from tensorboardX import SummaryWriter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import os
import gc
import torch
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
from dataloader import LossWrapper
from models import ProteinAttDataset, ProteinCrossAttentionModel, ProteinJax
from utils import write_information, save_metrics_to_csv, seed_torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score

parser = argparse.ArgumentParser(description='Predict Molecular Phenotypes')
parser.add_argument('--trials', type=int, default=1000, required=False, help='The trials that we are trying')
parser.add_argument('--epochs', type=int, default=30, required=False, help='Epochs to train the model')
parser.add_argument('--focal', type=bool, default=False, required=False, help='Use Focal Loss')
parser.add_argument('--outcome', type=str, default='function', required=False, help='The outcome that we are trying to predict')


dataTrain = pd.read_csv('dataset/data_train.csv', index_col='identifier')


def focal_loss(logits, labels, gamma=2.0):
    logits = logits.float()
    labels = labels.long() + 1

    ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
    p_t = torch.exp(-ce_loss)
    focal = (1 - p_t) ** gamma * ce_loss
    return focal.mean()


def collate_fn(batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wt, diff, outcome = zip(*batch)

    wt = [torch.tensor(x, device=device) for x in wt]
    diff = [torch.tensor(x, device=device) for x in diff]

    padded_wt = pad_sequence(wt, batch_first=True)
    padded_diff = pad_sequence(diff, batch_first=True)

    padded_batch = torch.stack([padded_wt, padded_diff], dim=1)
    outcome_tensor = torch.tensor(outcome, dtype=torch.long, device=device)
    return padded_batch, outcome_tensor


def train_test(args, train_loader, val_loader, writer, cv, device):
    model = ProteinCrossAttentionModel(input_dim=1280)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-3)
    weight = torch.tensor([1.23, 7.83, 16.60], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

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

        mcc_train, balanced_acc_train = 0, 0

        for features, outcome in train_loader:
            features, outcome = features.to(device), outcome.to(device)

            optimizer.zero_grad()
            out = model(features)

            probas = F.softmax(out, dim=1)
            preds = torch.argmax(probas, dim=1)

            if args.focal:
                preds = torch.argmax(probas, dim=1)
                loss = criterion(probas, outcome)
            else:
                logits = out.float()
                labels = outcome.long() + 1

                loss = criterion(logits, labels)
                del logits, labels
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            y_true_train.extend(outcome.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

            del out, probas, preds, loss
            torch.cuda.empty_cache()

        y_true_train = np.array(y_true_train)
        y_pred_train = np.array(y_pred_train)

        mcc_train = matthews_corrcoef(y_true_train, y_pred_train)
        balanced_acc_train = balanced_accuracy_score(y_true_train, y_pred_train)

        scheduler.step()
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            y_true_val, y_pred_val = [], []

            mcc_val, balanced_acc_val = 0, 0

            for features, outcome in val_loader:
                features, outcome = features.to(device), outcome.to(device)
                out = model(features)
                probas = F.softmax(out, dim=1)
                preds = torch.argmax(probas, dim=1)
                
                if args.focal:
                    preds = torch.argmax(probas, dim=1)
                    loss_val = criterion(probas, outcome)
                else:
                    logits = out.float()
                    labels = outcome.long() + 1

                    loss_val = criterion(logits, labels)
                    del logits, labels

                eval_loss += loss_val
                y_true_val.extend(outcome.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

                del out, probas, preds, loss_val
                torch.cuda.empty_cache()
            
            eval_final = eval_loss / len(val_loader)
            y_true_val = np.array(y_true_val)
            y_pred_val = np.array(y_pred_val)

            mcc_val = matthews_corrcoef(y_true_val, y_pred_val)
            balanced_acc_val = balanced_accuracy_score(y_true_val, y_pred_val)

        print('Train_loss: {:.4f}, Val_Loss: {:.4f}, MCC_train: {:.4f}, MCC_Val: {:.4f}\n'.format(total_loss, eval_final, mcc_train, mcc_val))
        
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

    writer.close()


def create_loader(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda:0':
        torch.cuda.empty_cache()
        gc.collect()
    print('Device:', device)

    seed_torch(seed=42, device=device)

    os.makedirs(os.path.join(folders['train_val']['trials'], str(args.trials)), exist_ok=True)
    os.makedirs(os.path.join(folders['train_val']['logs'], str(args.trials)), exist_ok=True)


    for cv, split in enumerate(os.listdir(folders['splits']['splits'])):
        writer = SummaryWriter(os.path.join(folders['train_val']['logs'], str(args.trials), f'cv_{cv}'), flush_secs=15)

        if split.endswith('.csv'):
            print(f'Cross Validation: Fold {cv}')
            split_data = pd.read_csv(f'{os.path.join(folders["splits"]["splits"], split)}')
            ids_train = list(split_data['train'])
            ids_val = list(split_data['val'].dropna())

            train_dataset = ProteinAttDataset(ids_train, dataTrain, folders['embedding']['mut'], folders['embedding']['diff'], args.outcome)
            val_dataset = ProteinAttDataset(ids_val, dataTrain, folders['embedding']['mut'], folders['embedding']['diff'], args.outcome)

            train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False, drop_last=True)

            train_test(args, train_loader, val_loader, writer, cv, device)


if __name__ == '__main__':
    args = parser.parse_args()
    create_loader(args)

    print("Done!")