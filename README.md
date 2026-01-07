# spectroscopy-fewshot-benchmark
A unified, task-oriented benchmark for few-shot regression in NIR/MIR spectroscopy, providing standardized preprocessing, task definitions, and reference implementations for transfer learning and meta-learning methods.

## Dataset Generation

To generate the complete benchmark dataset, follow these steps in order:

### 1. Download raw data

Run the `fetch_data.py` script to download all original datasets from their public sources:

```bash
python fetch_data.py
```

This script will automatically download the following datasets:
- **Mango**: Mango quality prediction data
- **Melamine**: Melamine adulteration dataset
- **Corn**: Corn properties (moisture, oil, protein, starch)
- **Diesel**: Diesel properties (BP50, CN, density, flash point, freeze point, total content, viscosity)
- **Eggs**: Eggs dataset
- **Wheat kernel**: Wheat kernel dataset
- **CGL**: CGL NIR dataset
- **NIR Shootout 2002**: NIR Shootout 2002 dataset
- **OSSL**: Open Soil Spectral Library (MIR and NIR soil data)

Files will be saved in the `data_tmp/` directory. Some compressed files (.zip, .gz) are automatically decompressed.

### 2. Transform soil data

OSSL soil data requires special processing. Run:

```bash
python transform_soil_data.py
```

This script:
- Reads the raw OSSL data (`data_tmp/ossl_all_L0_v1.2.csv`)
- Applies transformations and mappings defined in `data_base/`
- Generates two clean datasets:
  - `data_tmp/soil_data_transformed_nir.csv` (NIR soil data)
  - `data_tmp/soil_data_transformed_mir.csv` (MIR soil data)

### 3. Generate partitions

Finally, generate the train/validation partitions for all datasets:

```bash
python generate_partitions.py
```

This script:
- Reads the predefined partition indices from `data_base/`
- Processes all downloaded datasets
- Generates the final partitions in the `data/` directory
- Creates for each task the following files:
  - `X_supp.csv`: Support spectra
  - `y_supp.csv`: Support labels
  - `X_query.csv`: Query spectra
  - `y_query.csv`: Query labels

Optionally, you can process only a specific dataset:

```bash
python generate_partitions.py --dataset soil_mir --verbose
```

Available datasets: `diesel`, `corn`, `melamine`, `eggs`, `soil_nir`, `soil_mir`, `mango`, `cgl`, `shootout`, `wheat`, `raman`.

---
