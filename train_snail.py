# coding=utf-8
from models.snail import SnailFewShot
from datasets_snail import init_regression_dataset

import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import os
import copy
import pandas as pd

import json

from preprocessing import PreprocessNIR

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error


def init_model(opt):
    model = SnailFewShot(opt.shots, opt.queries)
    model = model.cuda() if opt.cuda else model
    return model

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def labels_to_one_hot(opt, labels):
    if opt.cuda:
        labels = labels.cpu()
    labels = labels.numpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs

def batch_for_few_shot_regression(opt, x, y, test=False, shots=None):
    if shots is None:
        if test:
            seq_size = opt.shots_test + 1
            if (x.shape[0] // seq_size) != opt.batch_size:
                seq_size = x.shape[0] // opt.batch_size
        else:
            seq_size = opt.shots + 1
    else:
        seq_size = shots + 1

    labels = []
    last_targets = []
    # if not test:
    #     for i in range(opt.batch_size):
    #         labels.append(y[i * seq_size: (i + 1) * seq_size])
    #         last_targets.append(y[(i + 1) * seq_size - 1])
    # else:
    # seq_size = opt.shots + 1
    for i in range(x.shape[0]//seq_size):
        labels.append(y[i * seq_size: (i + 1) * seq_size])
        last_targets.append(y[(i + 1) * seq_size - 1])
    last_targets = Variable(torch.Tensor(last_targets))
    labels = [torch.Tensor(temp) for temp in labels]
    y = torch.cat(labels, dim=0)
    x, y = Variable(x), Variable(y)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
        last_targets = last_targets.cuda()
    return x, y, last_targets.unsqueeze(1)

def batch_for_few_shot(opt, x, y):
    seq_size = opt.shots + opt.queries
    one_hots = []
    last_targets = []
    for i in range(opt.batch_size):
        one_hot, idxs = labels_to_one_hot(opt, y[i * seq_size: (i + 1) * seq_size])
        one_hots.append(one_hot)
        last_targets.append(idxs[-1])
    last_targets = Variable(torch.Tensor(last_targets).long())
    one_hots = [torch.Tensor(temp) for temp in one_hots]
    y = torch.cat(one_hots, dim=0)
    x, y = Variable(x), Variable(y)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
        last_targets = last_targets.cuda()
    return x, y, last_targets

def get_acc(last_model, last_targets):
    # calculate MAE
    return torch.mean(torch.abs(last_model - last_targets)).item()

def train(opt, tr_dataloader, model, optim, val_dataloader=None):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = np.inf
    history = {"train":[], "val":[]}

    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    last_model_path = os.path.join(opt.exp, 'last_model.pth')

    loss_fn = nn.MSELoss()

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda()
        # batch = next(tr_iter)
        task_count = 0
        for batch in tr_iter:
            if task_count == opt.task_batch:
                break
            for task in range(batch["features"].shape[0]): 
                if task_count == opt.task_batch:
                    break   
                optim.zero_grad()
                x, y = batch['features'][task], batch['targets'][task]
                x, y, last_targets = batch_for_few_shot_regression(opt, x, y)
                model_output = model(x, y)
                last_model = model_output[:, -1, :]
                loss = loss_fn(last_model, last_targets)
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(get_acc(last_model, last_targets))
                task_count += 1
        avg_loss = np.mean(train_loss[-batch["features"].shape[0]:])
        avg_acc = np.mean(train_acc[-batch["features"].shape[0]:])
        history["train"].append(avg_loss)
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            for task in range(batch["features"].shape[0]):
                x, y = batch['features'][task], batch['targets'][task]
                x, y, last_targets = batch_for_few_shot_regression(opt, x, y)
                model_output = model(x, y)
                last_model = model_output[:, -1, :]
                loss = loss_fn(last_model, last_targets)
                val_loss.append(loss.item())
                val_acc.append(get_acc(last_model, last_targets))
        avg_loss = np.mean(val_loss[-batch["features"].shape[0]:])
        avg_acc = np.mean(val_acc[-batch["features"].shape[0]:])
        postfix = ' (Best)' if avg_acc <= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc <= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])
        history["val"].append(avg_loss)

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc, history

def test(opt, test_dataset, model):
    model.eval()
    mses = []
    maes = []
    rmses = []
    r2s = []
    for task in test_dataset:
        batches, shots = task.sample_episode_fixed(opt.shots_test, opt.queries_test, batch_size=1)
        x, y = batches['features'], batches['targets']
        y_true = []
        y_pred = []
        for batch_idx in range(len(x)):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]        
            x_batch, y_batch, last_targets = batch_for_few_shot_regression(opt, x_batch, y_batch, test=True, shots=shots)
            model_output = model(x_batch, y_batch, shots=shots)
            last_model = model_output[:, -1, :]
            y_true.append(last_targets.cpu().detach().numpy())
            y_pred.append(last_model.cpu().detach().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
    print('Test MSE: {}, Test MAE: {}, Test RMSE: {}, Test R2: {}'.format(np.mean(mses), np.mean(maes), np.mean(rmses), np.mean(r2s)))
    results = pd.DataFrame({'mse_test': [np.mean(mses)], 'mae_test': [np.mean(maes)], 'rmse_test': [np.mean(rmses)], 'r2_test': [np.mean(r2s)]})
    return results

# def test(opt, test_dataset, model):
#     avg_acc = list()    
#     test_iter = iter(test_dataloader)
#     for batch in test_iter:
#         for task in range(batch["features"].shape[0]):
#             x, y = batch['features'][task], batch['targets'][task]
#             x, y, last_targets = batch_for_few_shot_regression(opt, x, y)
#             model_output = model(x, y)
#             last_model = model_output[:, -1, :]
#             avg_acc.append(get_acc(last_model, last_targets))
#     avg_acc = np.mean(avg_acc)
#     print('Test Acc: {}'.format(avg_acc))

#     return avg_acc

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

def main():
    '''
    Initialize everything and train
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default', help='output directory for results')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--iterations', type=int, default=10000, help='number of training iterations (unused)')
    parser.add_argument('--num_samples', type=int, default=1, help='number of samples per prediction')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--cuda', action='store_true', help='enable CUDA training')
    parser.add_argument('--dataset', type=str, default='TRIP', help='dataset name (TRIP, Mango_y, Mango_yr, Soil_MIR, Soil_NIR) or path')
    parser.add_argument('--shots', type=int, default=20, help='number of support samples per task during training')
    parser.add_argument('--queries', type=int, default=1, help='number of query samples per task during training')
    parser.add_argument('--shots_test', type=int, default=20, help='number of support samples per task at test time')
    parser.add_argument('--queries_test', type=int, default=1, help='number of query samples per task at test time')
    parser.add_argument('--repeats', type=int, default=1, help='number of experiment repetitions')
    parser.add_argument('--task_batch', type=int, default=20, help='number of tasks per batch')
    options = parser.parse_args()

    # Resolve dataset path
    options.dataset = resolve_dataset_path(options.dataset)

    if not os.path.exists(options.exp):
        os.makedirs(options.exp)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Save config
    os.makedirs(options.exp, exist_ok=True)
    with open(os.path.join(options.exp, "config.json"), "w") as f:
        json.dump(vars(options), f)

    results = []
    results_best = []
    for rep in range(options.repeats):
        preprocessor = PreprocessNIR(savgol=True, scale=False, scale_y=True, window_length=15, polyorder=2, deriv=1)

        tr_dataloader, val_dataloader, test_dataloader, tr_dataset, val_dataset, test_dataset = init_regression_dataset(options, preprocessor=preprocessor)
        model = init_model(options)
        optim = torch.optim.Adam(params=model.parameters(), lr=options.lr)
        res = train(opt=options,
                    tr_dataloader=tr_dataloader,
                    val_dataloader=val_dataloader,
                    model=model,
                    optim=optim)
        best_state, best_acc, train_loss, train_acc, val_loss, val_acc, history = res

        import matplotlib.pyplot as plt
        plt.plot(history["train"], label="train")
        plt.plot(history["val"], label="val")
        plt.legend()
        plt.savefig(os.path.join(options.exp, f'loss_rep-{rep}.png'))
        plt.close()

        print('Testing with last model..')
        results_rep = test(opt=options,
                    test_dataset=test_dataset,
                    model=model)
        results.append(results_rep)
        # results.to_csv(os.path.join(options.exp, 'results.csv'))

        model.load_state_dict(best_state)
        print('Testing with best model..')
        results_best_rep = test(opt=options,
            test_dataset=test_dataset,
            model=model)
        results_best.append(results_best_rep)
        # results_best.to_csv(os.path.join(options.exp, 'results_best.csv'))
    
    results = pd.concat(results)
    results_best = pd.concat(results_best)
    results.loc['mean'] = results.mean()
    results_best.loc['mean'] = results_best.mean()
    results.to_csv(os.path.join(options.exp, 'results.csv'))
    results_best.to_csv(os.path.join(options.exp, 'results_best.csv'))
    

if __name__ == '__main__':
    main()
