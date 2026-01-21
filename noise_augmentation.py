import torch
from scipy.signal import savgol_filter

# 1) Ruido aditivo gaussiano — controlado por SNR en dB
class AddGaussianNoiseSNR:
    """Añade AWGN para alcanzar un SNR objetivo (dB) por espectro."""
    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def __call__(self, x: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # x: (..., N)
        sig_rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-12)
        noise_rms = sig_rms / (10.0**(self.snr_db/20.0))
        n = torch.randn_like(x)
        n = n / (torch.sqrt(torch.mean(n**2, dim=-1, keepdim=True)) + 1e-12)  # RMS=1
        return x + noise_rms * n


# 2) Drift de instrumento — ganancia global + offset por espectro
class InstrumentDrift:
    """
    Aplica drift instrumental: x' = (1 + g) * x + o
      - g ~ N(0, sigma_gain)  (ganancia multiplicativa alrededor de 1)
      - o ~ N(0, sigma_offset * std(x))  (offset relativo a la energía del espectro)
    """
    def __init__(self, sigma_gain: float = 0.05, sigma_offset: float = 0.05, clip_gain: tuple = (0.6, 1.6)):
        self.sigma_gain = sigma_gain
        self.sigma_offset = sigma_offset
        self.clip_gain = clip_gain

    def __call__(self, x: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # Ganancia multiplicativa por espectro
        if self.sigma_gain > 0:
            g = torch.randn(*x.shape[:-1], 1, device=x.device, generator=generator) * self.sigma_gain
            gain = (1.0 + g).clamp(*self.clip_gain)
        else:
            gain = torch.ones(*x.shape[:-1], 1, device=x.device)

        # Offset aditivo por espectro, relativo a std(x)
        if self.sigma_offset > 0:
            stdx = torch.std(x, dim=-1, keepdim=True) + 1e-12
            o = torch.randn(*x.shape[:-1], 1, device=x.device, generator=generator) * (self.sigma_offset * stdx)
        else:
            o = torch.zeros(*x.shape[:-1], 1, device=x.device)

        return gain * x + o


# 3) Curvatura de baseline — polinomio de bajo orden a lo largo de λ
class BaselinePolynomial:
    """
    Suma un polinomio suave p(λ) de orden bajo (2–3) como baseline:
      p(λ) = c0 + c1*λ + c2*λ^2 + ...
    donde cada coeficiente c_k ~ N(0, coef_std * std(x)) por espectro.
    """
    def __init__(self, order: int = 2, coef_std: float = 0.05):
        assert order >= 1, "order debe ser >= 1"
        self.order = order
        self.coef_std = coef_std

    def __call__(self, x: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # x: (..., N)
        N = x.shape[-1]
        if self.coef_std <= 0 or N < 2:
            return x

        device = x.device
        pos = torch.linspace(0.0, 1.0, N, device=device)  # λ normalizada
        stdx = torch.std(x, dim=-1, keepdim=True) + 1e-12

        # Coeficientes por espectro: shape (..., 1)
        poly = torch.zeros_like(x)
        p = torch.ones(N, device=device)  # pos^0
        for k in range(self.order + 1):
            c = torch.randn(*x.shape[:-1], 1, device=device, generator=generator) * (self.coef_std * stdx)
            poly = poly + c * p  # broadcast en la última dimensión
            p = p * pos  # siguiente potencia

        return x + poly


# Combinador con probabilidades y (opcional) Savitzky–Golay
class NoiseAugmentation:
    """
    Aplica (con probabilidad) las tres degradaciones:
      - AWGN controlado por SNR (AddGaussianNoiseSNR)
      - Drift instrumental (InstrumentDrift)
      - Baseline polinómica (BaselinePolynomial)
    y opcionalmente filtra con Savitzky–Golay al final.
    """
    def __init__(self,
                 snr_db: float = 20.0,
                 sigma_gain: float = 0.05,
                 sigma_offset: float = 0.05,
                 poly_order: int = 2,
                 poly_coef_std: float = 0.02,
                 p_awgn: float = 1.0,
                 p_drift: float = 1.0,
                 p_baseline: float = 1.0,
                 savgol: bool = False,
                 window_length: int = 5,
                 polyorder: int = 2,
                 deriv: int = 0):
        self.transforms = [
            ("awgn", AddGaussianNoiseSNR(snr_db), p_awgn),
            ("drift", InstrumentDrift(sigma_gain, sigma_offset), p_drift),
            ("baseline", BaselinePolynomial(poly_order, poly_coef_std), p_baseline),
        ]
        self.savgol = savgol
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def set_probabilities(self, p_awgn: float = 1.0, p_drift: float = 1.0, p_baseline: float = 1.0):
        self.transforms[0] = (self.transforms[0][0], self.transforms[0][1], p_awgn)
        self.transforms[1] = (self.transforms[1][0], self.transforms[1][1], p_drift)
        self.transforms[2] = (self.transforms[2][0], self.transforms[2][1], p_baseline)

    def __call__(self, x: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # import copy
        # x_orig = copy.deepcopy(x)
        for _, transform, p in self.transforms:
            do = (p >= 1.0) or (torch.rand((), device=x.device, generator=generator).item() < p)
            if do:
                x = transform(x, generator=generator)

        if self.savgol:
            N = x.shape[-1]
            # asegurar window_length válido (impar y <= N)
            wl = min(self.window_length, N if N % 2 == 1 else N - 1)
            if wl < 3:
                return x
            if wl % 2 == 0:
                wl = max(3, wl - 1)
            arr = x.cpu().numpy()
            filt = savgol_filter(arr, window_length=wl,
                                 polyorder=self.polyorder, deriv=self.deriv,
                                 axis=-1)
            x = torch.tensor(filt, dtype=x.dtype, device=x.device)
            # filt_orig = savgol_filter(x_orig.detach().cpu().numpy(), window_length=wl,
            #                      polyorder=self.polyorder, deriv=self.deriv,
            #                      axis=-1)
            # x_orig = torch.tensor(filt_orig, dtype=x.dtype, device=x.device)
        return x
