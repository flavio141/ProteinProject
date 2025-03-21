from tensorboardX import SummaryWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import gc
import math
import torch
import shutil
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
from GNN import ComplexGATModel, GATModel
from utils import write_information, save_metrics_to_csv, seed_torch
from focal_loss.focal_loss import FocalLoss
from dataloader import GraphDataset, LossWrapper
from torch_geometric.loader import DataLoader

from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, auc, roc_curve

parser = argparse.ArgumentParser(description='Predict Molecular Phenotypes')
parser.add_argument('--trials', type=int, default=1000, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=50, required=False, help='Epochs to train the model')
parser.add_argument('--remove', type=bool, default=True, required=False, help='Remove the old trials')
parser.add_argument('--remove_logs', type=bool, default=True, required=False, help='Remove the old logs')
parser.add_argument('--multi_gpu', type=bool, default=False, required=False, help='Use multiple GPUs')
parser.add_argument('--focal', type=bool, default=True, required=False, help='Use Focal Loss')
parser.add_argument('--outcome', type=str, default='function', required=False, help='The outcome that we are trying to predict')


dataTrain = pd.read_csv('dataset/data_train.csv', index_col='identifier')


def train_test(args, train_loader, val_loader, writer, cv, device):

    model = ComplexGATModel(1280, 64, 1, heads=4)
    model.to(device)
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)#, weight_decay=1e-3)
    if args.focal:
        criterion = FocalLoss(gamma=2.0, ignore_index=-999)
    else:
        loss_function = nn.BCEWithLogitsLoss() 
        criterion = LossWrapper(loss_function, ignore_index=-999)


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1)

    seed_torch(seed=42, device=device)
    
    if int(cv) == 0:
        csv_folder = os.path.join(folders['train_val']['results'], 'trial_' + str(args.trials) + '.txt')
        write_information(args, csv_folder, model, optimizer, criterion, args.epochs, scheduler)

    model.train()
    for epoch in range(args.epochs):
        print('\nEpoch:', epoch + 1)
        total_loss = 0
        y_true_train, y_pred_train = [], []

        mcc_train, balanced_acc_train, auc_train, auroc_train = 0, 0, 0, 0

        for train_batch in tqdm(train_loader, desc='Training Batch'):
            train_batch = train_batch.to(device)
            if torch.all(train_batch.y == -999):
                continue

            optimizer.zero_grad()

            out = model(train_batch)
            if args.focal:
                m = nn.Sigmoid()
                loss = criterion(m(out.reshape(-1)), train_batch.y.long().view(-1,))
            else:
                loss = criterion(out, train_batch.y)

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

            for val_batch in tqdm(val_loader, desc='Validation Batch'):
                val_batch = val_batch.to(device)
                if torch.all(val_batch.y == -999):
                    continue
                
                out_val = model(val_batch)
                if args.focal:
                    m_out = nn.Sigmoid()
                    loss_val = criterion(m_out(out_val.reshape(-1)), val_batch.y.long().view(-1,))
                else:
                    loss_val = criterion(out_val, val_batch.y)

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

        print('Train_loss: {:.4f}, Val_Loss: {:.4f}, MCC_train: {:.4f}, MCC_Val: {:.4f}'.format(total_loss, eval_final, mcc_train, mcc_val))
        
        save_metrics_to_csv({'Loss_Train': total_loss, 
                             'Loss_Val': eval_final,
                             'MCC_Train': mcc_train, 
                             'MCC_Val': mcc_val}, epoch=epoch)

        writer.add_scalar('train/loss', total_loss, epoch)
        writer.add_scalar('val/loss', eval_final, epoch)

        writer.add_scalar('train/mcc', mcc_train, epoch)
        writer.add_scalar('val/mcc', mcc_val, epoch)

        writer.add_scalar('train/balanced_acc', balanced_acc_train, epoch)
        writer.add_scalar('val/balanced_acc', balanced_acc_val, epoch)

        writer.add_scalar('train/auc', auc_train, epoch)
        writer.add_scalar('val/auc', auc_score_val, epoch)

        writer.add_scalar('train/auroc', auroc_train, epoch)
        writer.add_scalar('val/auroc', auroc_val, epoch)

    writer.close()


def create_loader(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda:0':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    if os.path.exists(os.path.join(folders['train_val']['trials'], str(args.trials))) and args.remove:
        shutil.rmtree(os.path.join(folders['train_val']['trials'], str(args.trials)))
        os.makedirs(os.path.join(folders['train_val']['trials'], str(args.trials)))
    else:
        os.makedirs(os.path.join(folders['train_val']['trials'], str(args.trials)), exist_ok=True)


    if os.path.exists(os.path.join(folders['train_val']['logs'], str(args.trials))) and args.remove_logs:
        shutil.rmtree(os.path.join(folders['train_val']['logs'], str(args.trials)))
        os.makedirs(os.path.join(folders['train_val']['logs'], str(args.trials)))
    else:
        os.makedirs(os.path.join(folders['train_val']['logs'], str(args.trials)), exist_ok=True)


    for cv, split in enumerate(os.listdir(folders['splits']['splits'])):
        writer = SummaryWriter(os.path.join(folders['train_val']['logs'], str(args.trials), f'cv_{cv}'), flush_secs=15)

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

            train_test(args, train_loader, val_loader, writer, cv, device)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)

    print("Done!")