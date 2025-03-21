import os
import shutil
import argparse
import pandas as pd

import sys
sys.path.append('utils')

from env import folders, data
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser(description='Prepare Train Data')
parser.add_argument('--outcome', type=str, default='function', required=False, help='The outcome that we are trying to predict')

def create_split(args):
    data_modified = data.assign(mutation=data['mutation'].str.split(',')).explode('mutation')

    data_modified['identifier'] = data_modified['uniprot_id'] + '_' + data_modified['wildtype'] + '_' + data_modified['position'].astype(str) + '_' + data_modified['mutation']
    if args.outcome == 'function':
        data_modified['function'] = data_modified['function'].abs().fillna(-999)
    else:
        data_modified['interaction'] = data_modified['interaction'].abs().fillna(-999)

    file_to_keep = [file.split('.')[0] for file in os.listdir(folders['embedding']['graphs']) if file.split('.')[0] in list(data_modified['identifier'])]
    data_modified = data_modified[data_modified['identifier'].isin(file_to_keep)].drop_duplicates(subset='identifier', keep='first')
    data_modified.to_csv('dataset/data_train.csv')
    
    if args.outcome == 'function':
        labels = data_modified['function']
    else:
        labels = data_modified['interaction']

    if os.path.exists(folders['splits']['splits']):
        shutil.rmtree(folders['splits']['splits'])
    os.makedirs(folders['splits']['splits'])

    skf = StratifiedKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(skf.split(data_modified, labels)):
        train_df = list(data_modified.iloc[train_idx]['identifier'])
        val_df = list(data_modified.iloc[val_idx]['identifier'])
        val_df += [''] * (len(train_df) - len(val_df))

        split_df = pd.DataFrame({'train': train_df, 'val': val_df})
        split_df.to_csv(f'{os.path.join(folders["splits"]["splits"], str(fold))}.csv', index=False)

        if os.path.exists(f'{os.path.join(folders["splits"]["splits"], str(fold))}.csv'):
            print(f'File number {fold} created!')


if __name__ == '__main__':
    args = parser.parse_args()
    create_split(args)
