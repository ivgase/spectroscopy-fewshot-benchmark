import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import tqdm
import os
import shutil
import json
import copy
import pickle

from copy import deepcopy

from datasets_fewshot import MixedDataset
from models.maml import MAML
from models.cnn import NIR_CNN, ResNet1D, ResidualBlock1D
from models.utils.utils import *
from models.utils.configs import *
from preprocessing import PreprocessNIR
from noise_augmentation import NoiseAugmentation

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="results/default", help="results directory")
parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
parser.add_argument("--update_lr", type=float, default=0.01, help="update learning rate")
parser.add_argument("--meta_lr", type=float, default=0.001, help="meta learning rate")
parser.add_argument("--k_spt", type=int, default=25, help="k shot for support set")
parser.add_argument("--k_qry", type=int, default=25, help="k shot for query set")
parser.add_argument("--k_spt_test", type=int, default=None, help="k shot for support set for test")
parser.add_argument("--k_qry_test", type=int, default=None, help="k shot for query set for test")
parser.add_argument("--update_step", type=int, default=5, help="update steps")
parser.add_argument("--update_step_test", type=int, default=10, help="update steps for test")
parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping")
parser.add_argument("--second_order", action="store_true", help="use second order")
parser.add_argument("--repeats", type=int, default=1, help="number of repeats")
parser.add_argument("--load_weights", type=str, default=None, help="load weights from a previous model. Provide the path")
parser.add_argument("--dataset", type=str, default="data/MixedDataset", help="dataset path")
parser.add_argument("--task_batch", type=int, default=None, help="number of task batches")
parser.add_argument("--train_tasks", type=int, default=None, help="total number of tasks")
parser.add_argument("--noise_aug", action="store_true", help="use noise augmentation")
parser.add_argument("--p_awgn", type=float, default=1.0, help="probability of applying SNR noise augmentation")
parser.add_argument("--p_drift", type=float, default=1.0, help="probability of applying scale noise augmentation")
parser.add_argument("--p_baseline", type=float, default=1.0, help="probability of applying baseline noise augmentation")
parser.add_argument("--savgol", action="store_true", help="use Savitzky-Golay filter")
parser.add_argument("--savgol_noise", action="store_true", help="use Savitzky-Golay filter after noise augmentation")
parser.add_argument("--adapt_clean", action="store_true", help="adapt clean data to noise augmentation")
args = parser.parse_args()

episodes = args.episodes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save config
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "config.json"), "w") as f:
    json.dump(vars(args), f)

if args.k_spt_test is None:
    args.k_spt_test = args.k_spt
if args.k_qry_test is None:
    args.k_qry_test = args.k_qry

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = os.path.join("logs", args.output.replace("results/", "")+"_"+timestamp)
# os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)
writer.add_hparams(vars(args), {})

results_df = []
for rep in range(args.repeats):
    # learner = NIR_CNN().to(device)
    learner = ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes=1).to(device)
    # learner.apply(weights_init_he)
    maml = MAML(args, learner, rep, writer).to(device)

    if args.load_weights is not None:
        maml.set_state_dict(torch.load(args.load_weights))

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    preprocessor = PreprocessNIR(savgol=args.savgol, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
    noise_aug = NoiseAugmentation(savgol=args.savgol_noise, window_length=15, polyorder=2, deriv=1, snr_db=45) if args.noise_aug else None
    if noise_aug is not None:
        noise_aug.set_probabilities(p_awgn=args.p_awgn, p_drift=args.p_drift, p_baseline=args.p_baseline)

    # Load data
    train_data = MixedDataset(path=args.dataset, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, 
                              preprocessor=preprocessor, max_tasks=args.train_tasks, noise_aug=noise_aug)
    val_data = MixedDataset(path=args.dataset, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, 
                            preprocessor=preprocessor, noise_aug=noise_aug)
    if len(val_data) == 0:
        val_data = deepcopy(train_data)
    # val_data = copy.deepcopy(train_data)
    noise_aug_test = NoiseAugmentation(savgol=args.savgol_noise, window_length=15, polyorder=2, deriv=1, snr_db=45) if args.noise_aug else None
    if noise_aug is not None:
        if not args.adapt_clean:
            noise_aug_test.set_probabilities(p_awgn=args.p_awgn, p_drift=args.p_drift, p_baseline=args.p_baseline)
        else:
            noise_aug_test.set_probabilities(p_awgn=0.0, p_drift=0.0, p_baseline=0.0)
    test_data = MixedDataset(path=args.dataset, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, 
                             preprocessor=preprocessor, noise_aug=noise_aug_test)

    history = {"train": {"mse":{}, "mae":{}}, "val": {"mse":{}, "mae":{}}}
    for episode in range(args.episodes):
        if args.task_batch and args.task_batch < len(train_data):
            task_subset = random.sample(range(len(train_data)), args.task_batch)
        else:
            task_subset = None

        mses_train, maes_train, rmses_train, r2s_train = maml(train_data, save_weights=(episode == args.episodes-1),
                                                            task_subset=task_subset)
        # writer.add_scalar("MSE/Train", float(mses[-1]), episode)
        # writer.add_scalar("MAE/Train", float(maes[-1]), episode)
        # writer.add_scalar("RMSE/Train", float(rmses[-1]), episode)
        # writer.add_scalar("R2/Train", float(r2s[-1]), episode)
        history["train"]["mse"][episode] = float(mses_train[-1])
        history["train"]["mae"][episode] = float(maes_train[-1])

        if episode % 50 == 0:
            print(f"Episode {episode} | MSE: {mses_train[-1]:.4f} | MAE: {maes_train[-1]:.4f}")

        if episode % 250 == 0 or episode == args.episodes-1:
            mses = []
            maes = []
            rmses = []
            r2s = []
            full_mses = []
            full_maes = []
            full_rmses = []
            full_r2s = []

            preds_df_val = []
            for val_task in val_data:
                # data = val_task.sample(args.k_spt, args.k_qry)
                data = val_task.sample_fixed(args.k_spt_test, args.k_qry_test)
                x_supp = data["support_features"]
                y_supp = data["support_targets"]
                x_query = data["query_features"]
                y_query = data["query_targets"]

                net, fast_weights, mse, mae, rmse, r2 = maml.finetuning_batched(x_supp, y_supp, x_query, y_query, 
                                                                        batch_size=25, writer=None, task_name=val_task.name)
                # Save locally adapted model weights
                if args.episodes-1 == episode:
                    os.makedirs(os.path.join(args.output, "weights"), exist_ok=True)
                    pickle.dump(fast_weights, open(os.path.join(args.output, "weights", f"adapted_weights_val_{rep}_{val_task.name}.pkl"), "wb"))

                mses.append(mse)
                maes.append(mae)
                rmses.append(rmse)
                r2s.append(r2)

                y_true = []
                y_pred = []
                y_idx = []

                full_mse = 0.0
                full_mae = 0.0
                full_rmse = 0.0
                full_r2 = 0.0
                num_instances = 0
                val_dl = val_task.query_dataloader()
                for x, y, idx in val_dl:
                    y_true.extend(y.detach().numpy().squeeze(1).tolist())
                    x, y = x.to(device), y.to(device)
                    logits = net(x, vars=fast_weights)
                    full_mse += ((logits - y) ** 2).sum().item()
                    full_mae += torch.abs(logits - y).sum().detach().cpu()
                    num_instances += y.size(0)

                    y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
                    y_idx.extend(idx.detach().numpy().tolist())

                full_mse /= num_instances
                full_mae /= num_instances
                full_rmse = np.sqrt(full_mse)
                # full_r2 /= len(val_dl)
                full_mses.append(full_mse)
                full_maes.append(full_mae)
                full_rmses.append(full_rmse)
                # full_r2s.append(full_r2)
                full_r2 = r2_score(y_true, y_pred)
                full_r2s.append(full_r2)

                if episode == args.episodes-1:
                    preds_df_val.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": y_idx, "name": val_task.name}))

            if episode == args.episodes-1:
                # Save predictions
                preds_df_val = pd.concat(preds_df_val, axis=0).set_index("idx").sort_index()
                preds_df_val.to_csv(os.path.join(args.output, f"predictions_val_{rep}.csv"), index=True)

            mses = np.array(mses).mean(axis=0).astype(np.float16)
            maes = np.array(maes).mean(axis=0).astype(np.float16)
            rmses = np.array(rmses).mean(axis=0).astype(np.float16)
            r2s = np.array(r2s).mean(axis=0).astype(np.float16)
            full_mses = np.array(full_mses).mean().astype(np.float16)
            full_maes = np.array(full_maes).mean().astype(np.float16)
            full_rmses = np.array(full_rmses).mean().astype(np.float16)
            full_r2s = np.array(full_r2s).mean().astype(np.float16)

            history["val"]["mse"][episode] = full_mses
            history["val"]["mae"][episode] = full_maes

            # writer.add_scalar("MSE/Val", float(full_mses), episode)
            # writer.add_scalar("MAE/Val", float(full_maes), episode)
            # writer.add_scalar("RMSE/Val", float(full_rmses), episode)
            # writer.add_scalar("R2/Val", float(full_r2s), episode)

            print(f"Validation | MSE: {mses[-1]:.4f} | MAE: {maes[-1]:.4f} | Full MSE: {full_mses:.4f} | Full MAE: {full_maes:.4f}", flush=True)
            print(f"Validation | RMSE: {rmses[-1]:.4f} | R2: {r2s[-1]:.4f} | Full RMSE: {full_rmses:.4f} | Full R2: {full_r2s:.4f}", flush=True)

        # writer.add_scalars("MSE", {"Train": mses_train[-1], "Val": float(full_mses)}, episode)
        # writer.add_scalars("MAE", {"Train": maes_train[-1], "Val": float(full_maes)}, episode)
        # writer.add_scalars("RMSE", {"Train": rmses_train[-1], "Val": float(full_rmses)}, episode)
        # writer.add_scalars("R2", {"Train": r2s_train[-1], "Val": float(full_r2s)}, episode)

        writer.add_scalar(f"Metrics_rep-{rep}/MSE/Train", mses_train[-1], episode)
        writer.add_scalar(f"Metrics_rep-{rep}/MAE/Train", maes_train[-1], episode)
        writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Train", rmses_train[-1], episode)
        writer.add_scalar(f"Metrics_rep-{rep}/R2/Train", r2s_train[-1], episode)

        writer.add_scalar(f"Metrics_rep-{rep}/MSE/Val", float(full_mses), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/MAE/Val", float(full_maes), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Val", float(full_rmses), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/R2/Val", float(full_r2s), episode)

    # Save model
    torch.save(maml.get_state_dict(), os.path.join(args.output, f"model_{rep}.pth"))

    # Test model
    mses_test = []
    maes_test = []
    rmses_test = []
    r2s_test = []
    full_mses_test = []
    full_maes_test = []
    full_rmses_test = []
    full_r2s_test = []

    preds_df_test = []

    for test_task in test_data:
        data = test_task.sample_fixed(args.k_spt_test, args.k_qry_test)
        x_supp = data["support_features"]
        y_supp = data["support_targets"]
        x_query = data["query_features"]
        y_query = data["query_targets"]

        net, fast_weights, mse, mae, rmse, r2 = maml.finetuning_batched(x_supp, y_supp, x_query, y_query, 
                                                                        batch_size=25, writer=writer, task_name=test_task.name)
        # Save locally adapted model weights
        pickle.dump(fast_weights, open(os.path.join(args.output, "weights", f"adapted_weights_test_{rep}_{test_task.name}.pkl"), "wb"))

        mses_test.append(mse)
        maes_test.append(mae)
        rmses_test.append(rmse)
        r2s_test.append(r2)

        y_true = []
        y_pred = []
        y_idx = []

        full_mse = 0.0
        full_mae = 0.0
        full_rmse = 0.0
        full_r2 = 0.0
        num_instances = 0
        test_dl = test_task.query_dataloader()
        for x, y, idx in test_dl:
            x, y = x.to(device), y.to(device)
            logits = net(x, vars=fast_weights)
            full_mse += ((logits - y) ** 2).sum().item()
            full_mae += torch.abs(logits - y).sum().detach().cpu()
            num_instances += y.size(0)

            y_true.extend(y.detach().cpu().numpy().squeeze(1).tolist())
            y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
            y_idx.extend(idx.detach().numpy().tolist())

        full_mse /= num_instances
        full_mae /= num_instances
        full_rmse = np.sqrt(full_mse)
        full_mses_test.append(full_mse)
        full_maes_test.append(full_mae)
        full_rmses_test.append(full_rmse)
        # full_r2s_test.append(full_r2)
        full_r2 = r2_score(y_true, y_pred)
        full_r2s_test.append(full_r2)

        preds_df_test.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": y_idx, "name": test_task.name}))

    # Save predictions
    preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
    preds_df_test.to_csv(os.path.join(args.output, f"predictions_test_{rep}.csv"), index=True)


    mses_test = np.array(mses_test).mean(axis=0).astype(np.float16)
    maes_test = np.array(maes_test).mean(axis=0).astype(np.float16)
    rmses_test = np.array(rmses_test).mean(axis=0).astype(np.float16)
    r2s_test = np.array(r2s_test).mean(axis=0).astype(np.float16)
    full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
    full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
    full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
    full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

    print(f"Test | MSE: {mses_test[-1]:.4f} | MAE: {maes_test[-1]:.4f} | Full MSE: {full_mses_test:.4f} | Full MAE: {full_maes_test:.4f}", flush=True)
    print(f"Test | RMSE: {rmses_test[-1]:.4f} | R2: {r2s_test[-1]:.4f} | Full RMSE: {full_rmses_test:.4f} | Full R2: {full_r2s_test:.4f}", flush=True)

    results_df_rep = pd.DataFrame({"mse_val": [full_mses], "mse_test": [full_mses_test], "mae_val": [full_maes], "mae_test": [full_maes_test],
                                   "rmse_val": [full_rmses], "rmse_test": [full_rmses_test], "r2_val": [full_r2s], "r2_test": [full_r2s_test]}, index=[rep])
    results_df.append(results_df_rep)
    # results_df = pd.DataFrame({"mse": [full_mses, full_mses_test], "mae": [full_maes,full_maes_test]}, index=["Val", "Test"])
    # results_df.to_csv(os.path.join(args.output, "results.csv"), index=True)

    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_float(elem) for elem in obj]
        else:
            return obj

    history = convert_to_float(history)

    with open(os.path.join(args.output, f"history_{rep}.json"), "w") as f:
        json.dump(history, f)

    # Save model
    torch.save(maml.get_state_dict(), os.path.join(args.output, f"model_{rep}.pth"))
        
    # save history plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(history["train"]["mse"].keys(), history["train"]["mse"].values(), label="Train MSE")
    plt.plot(history["val"]["mse"].keys(), history["val"]["mse"].values(), label="Val MSE")
    plt.yscale("log")
    plt.xlabel("Episode")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(os.path.join(args.output, f"history_{rep}.png"))

    plt.figure(figsize=(12, 6))
    plt.plot(history["train"]["mse"].keys(), history["train"]["mae"].values(), label="Train MAE")
    plt.plot(history["val"]["mse"].keys(), history["val"]["mae"].values(), label="Val MAE")
    plt.yscale("log")
    plt.xlabel("Episode")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(os.path.join(args.output, f"history_mae_{rep}.png"))

results_df = pd.concat(results_df, axis=0)
# calculate mean by columns and add it as a row
results_df.loc["mean"] = results_df.mean(axis=0)
results_df.to_csv(os.path.join(args.output, "results.csv"), index=True)
writer.close()