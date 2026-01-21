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

from noise_augmentation import NoiseAugmentation

from copy import deepcopy

from datasets_fewshot import MixedDataset
from models.protonet import PrototypicalNetwork
from models.cnn import NIR_CNN, ResNet1D, ResidualBlock1D
from models.utils.utils import *
from models.utils.configs import *
from preprocessing import PreprocessNIR

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="results/default/", help="results directory")
parser.add_argument("--episodes", type=int, default=10000, help="number of episodes")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--k_spt", type=int, default=25, help="k shot for support set")
parser.add_argument("--k_qry", type=int, default=25, help="k shot for query set")
parser.add_argument("--k_spt_test", type=int, default=None, help="k shot for support set on test")
parser.add_argument("--k_qry_test", type=int, default=None, help="k shot for query set on test")
parser.add_argument("--repeats", type=int, default=1, help="number of repeats")
parser.add_argument("--embedding", type=str, default="after", help="embedding layer to use")
parser.add_argument("--dist_temp", type=float, default=0.5, help="distance temperature")
parser.add_argument("--load_weights", type=str, default=None, help="load weights from a previous model. Provide the path")
parser.add_argument("--dataset", type=str, default="data/MixedDataset", help="path to dataset")
parser.add_argument("--train_tasks", type=int, default=None, help="number of training tasks")
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

if args.k_spt_test is None:
    args.k_spt_test = args.k_spt
if args.k_qry_test is None:
    args.k_qry_test = args.k_qry

# Save config
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = os.path.join("logs", args.output.replace("results/", "")+"_"+timestamp)
# os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)
writer.add_hparams(vars(args), {})

results_df = []
for rep in range(args.repeats):
    # learner = NIR_CNN().to(device)
    learner = ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes=1).to(device)
    protonet = PrototypicalNetwork(learner=learner, device=device, meta_batch_size=1, 
                                   lr=args.lr, dist_temperature=args.dist_temp)

    if args.load_weights is not None:
        protonet.load_state(torch.load(args.load_weights))

    preprocessor = PreprocessNIR(savgol=args.savgol, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
    noise_aug = NoiseAugmentation(savgol=args.savgol_noise, window_length=15, polyorder=2, deriv=1, snr_db=45) if args.noise_aug else None
    if noise_aug is not None:
        noise_aug.set_probabilities(p_awgn=args.p_awgn, p_drift=args.p_drift, p_baseline=args.p_baseline)

    noise_aug_test = NoiseAugmentation(savgol=args.savgol_noise, window_length=15, polyorder=2, deriv=1, snr_db=45) if args.noise_aug else None
    if noise_aug_test is not None:
        if not args.adapt_clean:
            noise_aug_test.set_probabilities(p_awgn=args.p_awgn, p_drift=args.p_drift, p_baseline=args.p_baseline)
        else:
            noise_aug_test.set_probabilities(p_awgn=0.0, p_drift=0.0, p_baseline=0.0)


    # # Load data
    # train_data = SoilDataset(prop=props, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # val_data = SoilDataset(prop=props, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # test_data = SoilDataset(prop=props, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

    # Load data
    train_data = MixedDataset(path=args.dataset, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, 
                              preprocessor=preprocessor, max_tasks=args.train_tasks, noise_aug=noise_aug)
    val_data = MixedDataset(path=args.dataset, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, 
                            preprocessor=preprocessor, noise_aug=noise_aug)
    if len(val_data) == 0:
        val_data = deepcopy(train_data)
    test_data = MixedDataset(path=args.dataset, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, 
                             preprocessor=preprocessor, noise_aug=noise_aug)

    history = {"train": {"mse":{}, "mae":{}}, "val": {"mse":{}, "mae":{}}}
    for episode in range(args.episodes):
        task_num = len(train_data)
        mses = []
        maes = []
        rmses = []
        r2s = []
        for i in range(task_num):
            task = train_data[i]
            sampled_data = task.sample(args.k_spt, args.k_qry)
            x_spt = sampled_data['support_features']
            y_spt = sampled_data['support_targets']
            x_qry = sampled_data['query_features']
            y_qry = sampled_data['query_targets']

            mae, mse, rmse, r2, probs, preds = protonet.train(x_spt, y_spt, x_qry, y_qry, task_type='regression', embedding=args.embedding)
            mses.append(mse)
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)

        history["train"]["mse"][episode] = np.mean(mses)
        history["train"]["mae"][episode] = np.mean(maes)

        print(f"Episode {episode} | MSE: {np.mean(mses):.4f} | MAE: {np.mean(maes):.4f} | RMSE: {np.mean(rmses):.4f} | R2: {np.mean(r2s):.4f}", flush=True)

        if episode % 50 == 0 or episode == episodes-1:
            mses_val = []
            maes_val = []
            rmses_val = []
            r2s_val = []
            mses_full_val = []
            maes_full_val = []
            rmses_full_val = []
            r2s_full_val = []
            for val_task in val_data:
                sampled_data = val_task.sample_fixed(args.k_spt, args.k_qry)
                x_spt = sampled_data['support_features']
                y_spt = sampled_data['support_targets']
                x_qry = sampled_data['query_features']
                y_qry = sampled_data['query_targets']

                mae, mse, rmse, r2, probs, preds = protonet.evaluate(1, x_spt, y_spt, x_qry, y_qry, task_type='regression', embedding=args.embedding)
                mses_val.append(mse)
                maes_val.append(mae)
                rmses_val.append(rmse)
                r2s_val.append(r2)

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
                    y_np = y.detach().numpy()
                    y_true.extend(y_np.squeeze(1).tolist())
                    mae, mse, rmse, r2, probs, preds = protonet.evaluate(1, x_spt, y_spt, x, y, task_type='regression', embedding=args.embedding)
                    full_mse += ((preds - y_np) ** 2).sum().item()
                    full_mae += np.abs(preds - y_np).sum().item()
                    num_instances += y.size(0)

                    y_pred.extend(preds.squeeze(1).tolist())
                    y_idx.extend(idx.detach().numpy().tolist())                    
                    # full_mse += mse
                    # full_mae += mae
                    # full_rmse += rmse
                    # full_r2 += r2

                full_mse /= num_instances
                full_mae /= num_instances
                full_rmse = np.sqrt(full_mse)
                full_r2 = r2_score(y_true, y_pred)
                mses_full_val.append(full_mse)
                maes_full_val.append(full_mae)
                rmses_full_val.append(full_rmse)
                r2s_full_val.append(full_r2)

            mses_val = np.array(mses_val).mean(axis=0).astype(np.float16)
            maes_val = np.array(maes_val).mean(axis=0).astype(np.float16)
            rmses_val = np.array(rmses_val).mean(axis=0).astype(np.float16)
            r2s_val = np.array(r2s_val).mean(axis=0).astype(np.float16)
            mses_full_val = np.array(mses_full_val).mean().astype(np.float16)
            maes_full_val = np.array(maes_full_val).mean().astype(np.float16)
            rmses_full_val = np.array(rmses_full_val).mean().astype(np.float16)
            r2s_full_val = np.array(r2s_full_val).mean().astype(np.float16)

            history["val"]["mse"][episode] = mses_full_val
            history["val"]["mae"][episode] = maes_full_val

            print(f"Validation | MSE: {mses_val:.4f} | MAE: {maes_val:.4f} | Full MSE: {mses_full_val:.4f} | Full MAE: {maes_full_val:.4f}", flush=True)
            print(f"Validation | RMSE: {rmses_val:.4f} | R2: {r2s_val:.4f} | Full RMSE: {rmses_full_val:.4f} | Full R2: {r2s_full_val:.4f}", flush=True)

        writer.add_scalar(f"Metrics_rep-{rep}/MSE/Train", np.mean(mses), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/MAE/Train", np.mean(maes), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Train", np.mean(rmses), episode)
        writer.add_scalar(f"Metrics_rep-{rep}/R2/Train", np.mean(r2s), episode)

        writer.add_scalar(f"Metrics_rep-{rep}/MSE/Val", mses_full_val, episode)
        writer.add_scalar(f"Metrics_rep-{rep}/MAE/Val", maes_full_val, episode)
        writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Val", rmses_full_val, episode)
        writer.add_scalar(f"Metrics_rep-{rep}/R2/Val", r2s_full_val, episode)

    # Save model
    torch.save(protonet.dump_state(), os.path.join(args.output, f"model_{rep}.pth"))

    preds_df_test = []

    # Test model
    mses_test = []
    maes_test = []
    rmses_test = []
    r2s_test = []
    full_mses_test = []
    full_maes_test = []
    full_rmses_test = []
    full_r2s_test = []
    for test_task in test_data:
        data = test_task.sample_fixed(args.k_spt_test, args.k_qry_test)
        x_supp = data["support_features"]
        y_supp = data["support_targets"]
        x_query = data["query_features"]
        y_query = data["query_targets"]

        mae, mse, rmse, r2, probs, preds = protonet.evaluate(1, x_supp, y_supp, x_query, y_query, task_type="regression", embedding=args.embedding)
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
            y_np = y.detach().numpy()
            y_true.extend(y_np.squeeze(1).tolist())
            mae, mse, rmse, r2, probs, preds = protonet.evaluate(1, x_supp, y_supp, x, y, task_type='regression', embedding=args.embedding)
            full_mse += ((preds - y_np) ** 2).sum()
            full_mae += np.abs(preds - y_np).sum()
            num_instances += y.size(0)

            y_pred.extend(preds.squeeze(1).tolist())
            y_idx.extend(idx.detach().numpy().tolist())                    
            # full_mse += mse
            # full_mae += mae
            # full_rmse += rmse
            # full_r2 += r2

        full_mse /= num_instances
        full_mae /= num_instances
        full_rmse = np.sqrt(full_mse)
        full_r2 = r2_score(y_true, y_pred)
        full_mses_test.append(full_mse)
        full_maes_test.append(full_mae)
        full_rmses_test.append(full_rmse)
        full_r2s_test.append(full_r2)

        preds_df_test.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": y_idx, "name": test_task.name}))

    # Save predictions
    preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
    preds_df_test.to_csv(os.path.join(args.output, f"predictions_test_{rep}.csv"), index=True)

    mses_test = np.array(mses_test).mean().astype(np.float16)
    maes_test = np.array(maes_test).mean().astype(np.float16)
    rmses_test = np.array(rmses_test).mean().astype(np.float16)
    r2s_test = np.array(r2s_test).mean().astype(np.float16)
    full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
    full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
    full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
    full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

    print(f"Test | MSE: {mses_test:.4f} | MAE: {maes_test:.4f} | Full MSE: {full_mses_test:.4f} | Full MAE: {full_maes_test:.4f}", flush=True)
    print(f"Test | RMSE: {rmses_test:.4f} | R2: {r2s_test:.4f} | Full RMSE: {full_rmses_test:.4f} | Full R2: {full_r2s_test:.4f}", flush=True)

    results_df_rep = pd.DataFrame({"mse_val": [mses_full_val], "mse_test": [full_mses_test], "mae_val": [maes_full_val], "mae_test": [full_maes_test],
                                    "rmse_val": [rmses_full_val], "rmse_test": [full_rmses_test], "r2_val": [r2s_full_val], "r2_test": [full_r2s_test]}, index=[rep])
    results_df.append(results_df_rep)
    # results_df = pd.DataFrame({"mse": [mses_full_val, full_mses_test], "mae": [maes_full_val,full_maes_test]}, index=["Val", "Test"])
    # results_df.to_csv(os.path.join(args.output, "results.csv"), index=True)

    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_float(elem) for elem in obj]
        else:
            return obj

    history = convert_to_float(history)

    with open(os.path.join(args.output, f"history_{rep}.json"), "w") as f:
        json.dump(history, f)
        
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
    