import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import gc
import math
import torch
import optuna
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.append('utils')
sys.path.append('models')

from tqdm import tqdm
from env import folders
from GNN import GCNModel, ComplexGATModel
from optuna_utils import write_information, seed_torch, save_metrics_to_csv
from focal_loss.focal_loss import FocalLoss
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader

from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, auc, roc_curve

parser = argparse.ArgumentParser(description='Predict Molecular Phenotypes')
parser.add_argument('--epochs', type=int, default=50, required=False, help='Epochs to train the model')
parser.add_argument('--outcome', type=str, default='function', required=False, help='The outcome that we are trying to predict')


dataTrain = pd.read_csv('dataset/data_train.csv', index_col='identifier')


def train_test(args, parameters, train_loader, val_loader, cv, device, id):

    models = {
        'GCNModel': GCNModel(1280, 64, 1, dropout=parameters['dropout']),
        'ComplexGATModel': ComplexGATModel(1280, 64, 1, heads=4, dropout=parameters['dropout'])
    }
    model = models[parameters['model']]
    model.to(device)

    optimizers = {
        'AD': optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay']),
        'AW': optim.AdamW(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay']),
        'ASGD': optim.ASGD(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay']),
        'RMS': optim.RMSprop(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'], momentum=0.5),
        'SGD': optim.SGD(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'], momentum=0.3)
    }

    criterion = FocalLoss(gamma=parameters['gamma'], ignore_index=-999)
    optimizer = optimizers[parameters['optimizer_name']]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    
    if int(cv) == 0:
        files_folder = os.path.join('optuna', parameters['model'])
        write_information(os.path.join(files_folder, f'trial{str(id)}_information.txt'), model, optimizer, criterion, args)

    mccs = {}
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        y_true_train, y_pred_train = [], []

        mcc_train, balanced_acc_train, auc_train, auroc_train = 0, 0, 0, 0

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            if torch.all(train_batch.y == -999):
                continue

            optimizer.zero_grad()

            out = model(train_batch)
            m = nn.Sigmoid()
            loss = criterion(m(out.reshape(-1)), train_batch.y.long().view(-1,))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_true_train.extend(train_batch.y.cpu().numpy())
            y_pred_train.extend(F.sigmoid(out.cpu().detach()).numpy().reshape(-1,))
            torch.cuda.empty_cache()

        y_true_train = np.array(y_true_train)
        y_pred_train = np.array(y_pred_train)
        valid_indices = (y_true_train != -999)

        y_true_train = y_true_train[valid_indices]
        y_pred_train = (y_pred_train[valid_indices] > 0.5).astype(int)

        mcc_train = matthews_corrcoef(y_true_train, y_pred_train)
        balanced_acc_train = balanced_accuracy_score(y_true_train, y_pred_train)
        auc_train = roc_auc_score(y_true_train, y_pred_train)
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_pred_train)
        auroc_train = auc(fpr_train, tpr_train)

        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            y_true_val, y_pred_val = [], []

            mcc_val, balanced_acc_val, auc_score_val, auroc_val = 0, 0, 0, 0

            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                if torch.all(val_batch.y == -999):
                    continue
                
                out_val = model(val_batch)
                m_out = nn.Sigmoid()
                loss_val = criterion(m_out(out_val.reshape(-1)), val_batch.y.long().view(-1,))

                eval_loss += loss_val
                y_true_val.extend(val_batch.y.cpu().numpy())
                y_pred_val.extend(F.sigmoid(out_val.cpu().detach()).numpy().reshape(-1,))

                torch.cuda.empty_cache()
            
            eval_final = eval_loss / len(val_loader)
            y_true_val = np.array(y_true_val)
            y_pred_val = np.array(y_pred_val)
            valid_indices_val = (y_true_val != -999)

            y_true_val = y_true_val[valid_indices_val]
            y_pred_val = (y_pred_val[valid_indices_val] > 0.5).astype(int)

            mcc_val = matthews_corrcoef(y_true_val, y_pred_val)
            balanced_acc_val = balanced_accuracy_score(y_true_val, y_pred_val)
            auc_score_val = roc_auc_score(y_true_val, y_pred_val)
            fpr_val, tpr_val, _ = roc_curve(y_true_val, y_pred_val)
            auroc_val = auc(fpr_val, tpr_val)

            mccs[epoch] = mcc_val


        if (epoch + 1) % 5 == 0:
            print('Train_loss: {:.4f}, Val_Loss: {:.4f}, MCC_train: {:.4f}, MCC_Val: {:.4f}'.format(total_loss, eval_final, mcc_train, mcc_val))
        
        save_metrics_to_csv({'Loss_Train': total_loss, 
                             'Loss_Val': eval_final,
                             'MCC_Train': mcc_train, 
                             'MCC_Val': mcc_val,
                             'BAC_Train': balanced_acc_train,
                             'BAC_Val': balanced_acc_val,
                             'AUC_Train': auc_train,
                             'AUC_Val': auc_score_val,
                             'AUROC_Train': auroc_train,
                             'AUROC_Val': auroc_val}, parameters=parameters, epoch=epoch, trial_id=id, cv=cv)
    
    return np.mean(list(mccs.values())[-3:]) 


def optuna_loader(args, parameters, id):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda:0':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    if not os.path.exists(os.path.join('optuna', parameters['model'])):
        os.makedirs(os.path.join('optuna', parameters['model']), exist_ok=True)

    mccs = {}
    for cv, split in enumerate(os.listdir(folders['splits']['splits'])):
        if split.endswith('.csv'):
            print(f'Cross Validation: Fold {cv}')
            split_data = pd.read_csv(f'{os.path.join(folders["splits"]["splits"], split)}')
            ids_train = list(split_data['train'])
            ids_val = [x for x in list(split_data['val']) if not (isinstance(x, float) and math.isnan(x))]

            if args.outcome == 'function':
                outcome_train = dataTrain.loc[ids_train]['function'].tolist()
                outcome_val = dataTrain.loc[ids_val]['function'].tolist()
            else:
                outcome_train = dataTrain.loc[ids_train]['interaction'].tolist()
                outcome_val = dataTrain.loc[ids_val]['interaction'].tolist()

            train_dataset = GraphDataset(folders['embedding']['graphs'], ids_train, outcome_train)
            val_dataset = GraphDataset(folders['embedding']['graphs'], ids_val, outcome_val)

            train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=128, num_workers=8, shuffle=False, drop_last=True)

            mccs[cv] = train_test(args, parameters, train_loader, val_loader, cv, device, id)
    
    return mccs


def objective(trial):
    model = trial.suggest_categorical('model', ['ComplexGATModel', 'GCNModel'])
    lr = trial.suggest_categorical('lr', [1e-5, 1e-4, 1e-3, 1e-2])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    optimizer_name = trial.suggest_categorical('optimizer_name', ['AD', 'AW', 'ASGD', 'RMS', 'SGD'])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5])
    gamma = trial.suggest_int('gamma', 2, 5)

    args = parser.parse_args()
    parameters = {
        'model': model,
        'lr': lr,
        'weight_decay': weight_decay,
        'optimizer_name': optimizer_name,
        'dropout': dropout,
        'gamma': gamma
    }

    mccs = optuna_loader(args, parameters, trial._trial_id)
    return np.mean(list(mccs.values()))


if __name__ == '__main__':
    storage = "sqlite:///optuna_protein.db"
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        study_name='OptunaProtein',
        storage=storage,
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=100) # type: ignore
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))