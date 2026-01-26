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
parser.add_argument("--epochs", type=int, default=10000, help="number of adaptation epochs per task")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate for task adaptation")
parser.add_argument("--k_spt", type=int, default=25, help="number of support samples per task")
parser.add_argument("--k_qry", type=int, default=25, help="number of query samples per task")
parser.add_argument("--repeats", type=int, default=1, help="number of experiment repetitions")
parser.add_argument("--load_weights", type=str, default=None, help="path to pretrained model weights")
parser.add_argument("--dataset", type=str, default="TRIP", help="dataset name (TRIP, Mango_y, Mango_yr, Soil_MIR, Soil_NIR) or path")
args = parser.parse_args()

# Resolve dataset path
args.dataset = resolve_dataset_path(args.dataset)

epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save config
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "config.json"), "w") as f:
    json.dump(vars(args), f)

results_df = []
for rep in range(args.repeats):
    # tmp = filter(lambda x: x.requires_grad, learner.parameters())
    # num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print(learner)
    # print('Total trainable tensors:', num)

    preprocessor = PreprocessNIR(savgol=True, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)
    
    # # Load data
    # train_data = SoilDataset(prop=props, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # val_data = SoilDataset(prop=props, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # # val_data = copy.deepcopy(train_data)
    # test_data = SoilDataset(prop=props, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

    # Load data
    # train_data = MixedDataset(path=args.dataset, split='train', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # val_data = MixedDataset(path=args.dataset, split='val', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)
    # val_data = copy.deepcopy(train_data)
    test_data = MixedDataset(path=args.dataset, split='test', supp_sz=args.k_spt, query_sz=args.k_qry, preprocessor=preprocessor)

    preds_df_test = []

    mses = []
    maes = []
    rmses = []
    r2s = []
    task_num = len(test_data)
    for i in range(task_num):
        learner = ResNet1D(ResidualBlock1D, [2, 2, 2, 2], num_classes=1).to(device)

        if args.load_weights is not None:
            learner.load_state_dict(torch.load(args.load_weights))

        optimizer = Adam(learner.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()

        task = test_data[i]
        print("Training on task", task.name, flush=True)
        sampled_data = task.sample_fixed(args.k_spt, args.k_qry)
        x_spt = sampled_data['support_features']
        y_spt = sampled_data['support_targets']
        x_qry = sampled_data['query_features']
        y_qry = sampled_data['query_targets']

        history = {"train": {"mse":{}, "mae":{}}, "val": {"mse":{}, "mae":{}}}
        query_dl = task.query_dataloader()
        for epoch in range(epochs):
            learner.train()
            optimizer.zero_grad()
            logits = learner(x_spt)
            loss = loss_fn(logits, y_spt)
            loss.backward()
            optimizer.step()
            
            if epoch % 15 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}", flush=True)

            history["train"]["mse"][epoch] = loss.item()
            history["train"]["mae"][epoch] = torch.abs(logits - y_spt).mean().item()

            mse = 0.0
            mae = 0.0
            num_instances = 0
            for x, y, _ in query_dl:
                with torch.no_grad():
                    x = x.to(device)
                    y = y.to(device)
                    learner.eval()
                    logits = learner(x)
                    mse += ((logits - y) ** 2).sum().item()
                    mae += torch.abs(logits - y).sum().item()
                    num_instances += y.size(0)

            mse /= num_instances
            mae /= num_instances

            history["val"]["mse"][epoch] = mse
            history["val"]["mae"][epoch] = mae
        
        y_true = []
        y_pred = []
        y_idx = []
        mse = 0.0
        mae = 0.0
        num_instances = 0
        for x, y, idx in query_dl:
            with torch.no_grad():
                y_true.extend(y.detach().numpy().squeeze(1).tolist())
                x = x.to(device)
                y = y.to(device)
                learner.eval()
                logits = learner(x)
                mse += ((logits - y) ** 2).sum().item()
                mae += torch.abs(logits - y).sum().item()
                num_instances += y.size(0)
                y_pred.extend(logits.detach().cpu().numpy().squeeze(1).tolist())
                y_idx.extend(idx.detach().numpy().tolist())

        mse /= num_instances
        mae /= num_instances        
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

        preds_df_test.append(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": y_idx, "name": task.name}))

        # Save plot
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(args.output, task.name), exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(list(history["train"]["mse"].keys()), list(history["train"]["mse"].values()), label="Train")
        plt.plot(list(history["val"]["mse"].keys()), list(history["val"]["mse"].values()), label="Val")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(os.path.join(args.output, task.name, f"history_{rep}.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(list(history["train"]["mae"].keys()), list(history["train"]["mae"].values()), label="Train")
        plt.plot(list(history["val"]["mae"].keys()), list(history["val"]["mae"].values()), label="Val")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.savefig(os.path.join(args.output, task.name, f"history_mae_{rep}.png"))
        plt.close()

        # Save history
        with open(os.path.join(args.output, task.name, f"history_{rep}.json"), "w") as f:
            json.dump(history, f)

        # Save model
        torch.save(learner.state_dict(), os.path.join(args.output, task.name, f"model_{rep}.pt"))

    results_df.append({
        "mse": np.mean(mses),
        "mae": np.mean(maes),
        "rmse": np.mean(rmses),
        "r2": np.mean(r2s)
    })

    # Save predictions
    preds_df_test = pd.concat(preds_df_test, axis=0).set_index("idx").sort_index()
    preds_df_test.to_csv(os.path.join(args.output, f"predictions_test_{rep}.csv"), index=True)

results_df = pd.DataFrame(results_df)
results_df.loc["mean"] = results_df.mean(axis=0)
results_df.to_csv(os.path.join(args.output, "results.csv"))