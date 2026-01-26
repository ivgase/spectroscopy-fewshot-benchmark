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

from datasets_fewshot import MixedDataset
from models.maml import MAML
from models.cnn import NIR_CNN, ResidualBlock1D, ResNet1D
from models.utils.utils import *
from models.utils.configs import *
from preprocessing import PreprocessNIR
from sklearn.metrics import r2_score

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import argparse

# Dataset name to path mapping
DATASET_PATHS = {
    "TRIP": "data/TRIP",
    "Mango_y": "data/MangoDataset_by_year",
    "Mango_yr": "data/MangoDataset_by_year-region",
    "Soil_MIR": "data/SoilDataset_MIR",
    "Soil_NIR": "data/SoilDataset_NIR",
}

def resolve_dataset_path(dataset_arg):
    """Resolve dataset name to path. If name is in mapping, return path; otherwise assume it's already a path."""
    return DATASET_PATHS.get(dataset_arg, dataset_arg)

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="results/default/", help="output directory for results")
parser.add_argument("--epochs", type=int, default=10000, help="number of training epochs")
parser.add_argument("--epochs_adapt", type=int, default=10, help="number of epochs for test-time adaptation")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate for training")
parser.add_argument("--lr_adapt", type=float, default=0.01, help="learning rate for test-time adaptation")
parser.add_argument("--k_spt", type=int, default=25, help="number of support samples per task")
parser.add_argument("--k_qry", type=int, default=25, help="number of query samples per task")
parser.add_argument("--grad_clip", type=float, default=1.0, help="gradient clipping value")
parser.add_argument("--repeats", type=int, default=1, help="number of experiment repetitions")
parser.add_argument("--load_weights", type=str, default=None, help="path to pretrained model weights")
parser.add_argument("--dataset", type=str, default="TRIP", help="dataset name (TRIP, Mango_y, Mango_yr, Soil_MIR, Soil_NIR) or path")
args = parser.parse_args()

# Resolve dataset path
args.dataset = resolve_dataset_path(args.dataset)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = os.path.join("logs", args.output.replace("results/", "")+"_"+timestamp)
# os.makedirs(logdir, exist_ok=True)

writer = SummaryWriter(log_dir=logdir)
writer.add_hparams(vars(args), {})


def finetunning(model, optimizer, x_supp, y_supp, x_query, y_query, batch_size=32, task_name=None, rep=0):
    fast_weights = copy.deepcopy(model)
    fast_weights.train()
    optimizer = Adam(fast_weights.parameters(), lr=args.lr_adapt)
    loss_fn = nn.MSELoss()
    
    support_dataset = torch.utils.data.TensorDataset(x_supp, y_supp)
    support_loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
    
    for i in range(args.epochs_adapt):
        for x_batch, y_batch in support_loader:
            logits = fast_weights(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            fast_weights.eval()
            logits = fast_weights(x_query)
            mse = loss_fn(logits, y_query).item()
            mae = torch.abs(logits - y_query).mean().item()
            rmse = torch.sqrt(loss_fn(logits, y_query)).item()
            r2 = r2_score(y_query.detach().cpu().numpy(), logits.detach().cpu().numpy())

            if task_name is not None:
                writer.add_scalar(f"Adaptation_rep-{rep}/{task_name}/MSE", mse, i)
                writer.add_scalar(f"Adaptation_rep-{rep}/{task_name}/MAE", mae, i)
                writer.add_scalar(f"Adaptation_rep-{rep}/{task_name}/RMSE", rmse, i)
                writer.add_scalar(f"Adaptation_rep-{rep}/{task_name}/R2", r2, i)
    return fast_weights, mse, mae, rmse, r2

epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save config
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "config.json"), "w") as f:
    json.dump(vars(args), f)

results_df = []
for rep in range(args.repeats):
    # learner = NIR_CNN().to(device)
    learner = ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes=1).to(device)

    if args.load_weights is not None:
        learner.load_state_dict(torch.load(args.load_weights))

    optimizer = Adam(learner.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    tmp = filter(lambda x: x.requires_grad, learner.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(learner)
    print('Total trainable tensors:', num)

    preprocessor = PreprocessNIR(savgol=True, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
    
    # # Load data
    # train_data = SoilDataset(prop=props, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # val_data = SoilDataset(prop=props, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # # val_data = copy.deepcopy(train_data)
    # test_data = SoilDataset(prop=props, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

    # Load data
    train_data = MixedDataset(path=args.dataset, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    val_data = MixedDataset(path=args.dataset, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    if len(val_data) == 0:
        val_data = copy.deepcopy(train_data)
    test_data = MixedDataset(path=args.dataset, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

    train_supp = train_data.batch_dataloader(mode='support')
    train_query = train_data.batch_dataloader(mode='query')

    history = {"train": {"mse":{}, "mae":{}}, "val": {"mse":{}, "mae":{}}}
    for epoch in range(epochs):
        learner.train()
        mae_loss = 0.0
        mse_loss = 0.0
        rmse_loss = 0.0
        r2_loss = 0.0
        num_instances = 0
        y_true = []
        y_pred = []
        for i, (x, y) in enumerate(train_supp):
            y_true.extend(y.detach().cpu().numpy().squeeze(1).tolist())
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = learner(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            # mse_loss += loss.item()
            with torch.no_grad():
                mse_loss += ((logits - y) ** 2).sum().item()
                mae_loss += torch.abs(logits - y).sum().item()
                num_instances += y.size(0)
            y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())

        mae_loss /= num_instances
        mse_loss /= num_instances
        rmse_loss = np.sqrt(mse_loss)
        r2_loss = r2_score(y_true, y_pred)
        # r2_loss /= len(train_supp)

        print(f"Epoch {epoch} | MSE: {mse_loss:.4f} | MAE: {mae_loss:.4f} | R2: {r2_loss:.4f}")

        writer.add_scalar(f"Metrics_rep-{rep}/MSE/Train", mse_loss, epoch)
        writer.add_scalar(f"Metrics_rep-{rep}/MAE/Train", mae_loss, epoch)
        writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Train", rmse_loss, epoch)
        writer.add_scalar(f"Metrics_rep-{rep}/R2/Train", r2_loss, epoch)
                
        if epoch % 10 == 0 or epoch == epochs - 1:
            learner.eval()
            mae_loss = 0.0
            mse_loss = 0.0
            rmse_loss = 0.0
            r2_loss = 0.0
            num_instances = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for i, (x, y) in enumerate(train_query):
                    y_true.extend(y.detach().numpy().squeeze(1).tolist())
                    x, y = x.to(device), y.to(device)
                    logits = learner(x)
                    loss = loss_fn(logits, y)

                    mse_loss += ((logits - y) ** 2).sum().item()
                    mae_loss += torch.abs(logits - y).sum().item()
                    rmse_loss += torch.sqrt(loss).item()
                    # r2_loss += r2_score(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
                    num_instances += y.size(0)

                    # y_true.extend(y.detach().cpu().numpy().squeeze(1).tolist())
                    y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())

            mae_loss /= num_instances
            mse_loss /= num_instances
            rmse_loss = np.sqrt(mse_loss)
            r2_loss = r2_score(y_true, y_pred)

            print(f"Validation | MSE: {mse_loss:.4f} | MAE: {mae_loss:.4f} | R2: {r2_loss:.4f}", flush=True)
            history["train"]["mse"][epoch] = mse_loss
            history["train"]["mae"][epoch] = mae_loss

        if epoch % 50 == 0 or epoch == epochs - 1:
            mses = []
            maes = []
            rmses = []
            r2s = []
            full_mses = []
            full_maes = []
            full_rmses = []
            full_r2s = []
            for val_task in val_data:
                fast_weights = copy.deepcopy(learner)
                fast_weights.train()
                optimizer_inner = Adam(fast_weights.parameters(), lr=args.lr_adapt)

                data = val_task.sample_fixed(args.k_spt, args.k_qry)
                x_supp = data["support_features"]
                y_supp = data["support_targets"]
                x_query = data["query_features"]
                y_query = data["query_targets"]
                
                fast_weights, mse, mae, rmse, r2 = finetunning(fast_weights, optimizer_inner, x_supp, y_supp, x_query, y_query, task_name=None)
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
                    logits = fast_weights(x)
                    full_mse += ((logits - y) ** 2).sum().item()
                    full_mae += torch.abs(logits - y).sum().item()
                    num_instances += y.size(0)

                    y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
                    y_idx.extend(idx.detach().numpy().tolist())
                    # with torch.no_grad():
                    #     full_rmse += torch.sqrt(loss_fn(logits, y)).item()
                    #     full_r2 += r2_score(y.detach().cpu().numpy(), logits.detach().cpu().numpy())

                # full_mse /= len(val_dl)
                # full_mae /= len(val_dl)
                # full_rmse /= len(val_dl)
                # full_r2 /= len(val_dl)
                full_mse /= num_instances
                full_mae /= num_instances
                full_rmse = np.sqrt(full_mse)
                full_r2 = r2_score(y_true, y_pred)
                full_mses.append(full_mse)
                full_maes.append(full_mae)
                full_rmses.append(full_rmse)
                full_r2s.append(full_r2)

            mses = np.array(mses).mean().astype(np.float16)
            maes = np.array(maes).mean().astype(np.float16)
            rmses = np.array(rmses).mean().astype(np.float16)
            r2s = np.array(r2s).mean().astype(np.float16)
            full_mses = np.array(full_mses).mean().astype(np.float16)
            full_maes = np.array(full_maes).mean().astype(np.float16)
            full_rmses = np.array(full_rmses).mean().astype(np.float16)
            full_r2s = np.array(full_r2s).mean().astype(np.float16)

            history["val"]["mse"][epoch] = full_mses
            history["val"]["mae"][epoch] = full_maes
            
            print(f"Validation Tasks | MSE: {mses:.4f} | MAE: {maes:.4f} | Full MSE: {full_mses:.4f} | Full MAE: {full_maes:.4f}", flush=True)
            print(f"Validation Tasks | RMSE: {rmses:.4f} | R2: {r2s:.4f} | Full RMSE: {full_rmses:.4f} | Full R2: {full_r2s:.4f}", flush=True)

            writer.add_scalar(f"Metrics_rep-{rep}/MSE/Val", full_mses, epoch)
            writer.add_scalar(f"Metrics_rep-{rep}/MAE/Val", full_maes, epoch)
            writer.add_scalar(f"Metrics_rep-{rep}/RMSE/Val", full_rmses, epoch)
            writer.add_scalar(f"Metrics_rep-{rep}/R2/Val", full_r2s, epoch)

    # save model
    torch.save(learner.state_dict(), os.path.join(args.output, f"model_{rep}.pt"))

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
        fast_weights = copy.deepcopy(learner)
        fast_weights.train()
        optimizer_inner = Adam(fast_weights.parameters(), lr=args.lr_adapt)

        data = test_task.sample_fixed(args.k_spt, args.k_qry)
        x_supp = data["support_features"]
        y_supp = data["support_targets"]
        x_query = data["query_features"]
        y_query = data["query_targets"]

        fast_weights, mse, mae, rmse, r2 = finetunning(fast_weights, optimizer_inner, x_supp, y_supp, x_query, y_query, task_name=test_task.name, rep=rep)
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
            y_true.extend(y.detach().numpy().squeeze(1).tolist())
            x, y = x.to(device), y.to(device)
            logits = fast_weights(x)
            full_mse += ((logits - y) ** 2).sum().item()
            full_mae += torch.abs(logits - y).sum().detach().cpu()
            num_instances += y.size(0)
            
            y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
            y_idx.extend(idx.detach().numpy().tolist())

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

    mses_test = np.array(mses_test).mean(axis=0).astype(np.float16)
    maes_test = np.array(maes_test).mean(axis=0).astype(np.float16)
    rmses_test = np.array(rmses_test).mean(axis=0).astype(np.float16)
    r2s_test = np.array(r2s_test).mean(axis=0).astype(np.float16)
    full_mses_test = np.array(full_mses_test).mean().astype(np.float16)
    full_maes_test = np.array(full_maes_test).mean().astype(np.float16)
    full_rmses_test = np.array(full_rmses_test).mean().astype(np.float16)
    full_r2s_test = np.array(full_r2s_test).mean().astype(np.float16)

    print(f"Test | MSE: {mses_test:.4f} | MAE: {maes_test:.4f} | Full MSE: {full_mses_test:.4f} | Full MAE: {full_maes_test:.4f}", flush=True)
    print(f"Test | RMSE: {rmses_test:.4f} | R2: {r2s_test:.4f} | Full RMSE: {full_rmses_test:.4f} | Full R2: {full_r2s_test:.4f}", flush=True)

    results_df_rep = pd.DataFrame({"mse_val": [full_mses], "mse_test": [full_mses_test], "mae_val": [full_maes], "mae_test": [full_maes_test], 
                                   "rmse_val": [full_rmses], "rmse_test": [full_rmses_test], "r2_val": [full_r2s], "r2_test": [full_r2s_test]}, index=[rep])
    results_df.append(results_df_rep)
    # results_df = pd.DataFrame({"mse": [full_mses, full_mses_test], "mae": [full_maes,full_maes_test]}, index=["Val", "Test"])
    # results_df.to_csv(os.path.join(args.output, "results.csv"), index=True)

    # Save history
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

    # Save plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(list(history["train"]["mse"].keys()), list(history["train"]["mse"].values()), label="Train")
    plt.plot(list(history["val"]["mse"].keys()), list(history["val"]["mse"].values()), label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig(os.path.join(args.output, f"history_{rep}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(list(history["train"]["mae"].keys()), list(history["train"]["mae"].values()), label="Train")
    plt.plot(list(history["val"]["mae"].keys()), list(history["val"]["mae"].values()), label="Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig(os.path.join(args.output, f"history_mae_{rep}.png"))
    plt.close()

results_df = pd.concat(results_df, axis=0)
# calculate mean by columns and add it as a row
results_df.loc["mean"] = results_df.mean(axis=0)
results_df.to_csv(os.path.join(args.output, "results.csv"), index=True)