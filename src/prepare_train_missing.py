import os
import shutil
import pandas as pd
import sys
from sklearn.model_selection import StratifiedGroupKFold

sys.path.append('utils')
from env import folders, data

label_cols = ['interaction', 'binding', 'function', 'phosphorylation', 'ubiquination', 
              'glycosylation', 'sumoylation', 'acetylation', 'other_ptm', 'methylation', 
              'ribosylation', 'subcell_localization_change', 'mimetic_ptm', 'folding', 'expression']

def create_split():
    data_modified = data.assign(mutation=data['mutation'].str.split(',')).explode('mutation')
    
    data_modified['identifier'] = data_modified['uniprot_id'] + '_' + data_modified['wildtype'] + '_' + \
                                  data_modified['position'].astype(str) + '_' + data_modified['mutation']
    
    file_to_keep = [file.split('.')[0] for file in os.listdir(folders['embedding']['mut']) 
                    if file.split('.')[0] in list(data_modified['identifier'])]
    
    data_modified = data_modified[data_modified['identifier'].isin(file_to_keep)].drop_duplicates(subset='identifier', keep='first')
    data_modified[label_cols] = data_modified[label_cols].abs()
    data_modified[label_cols] = data_modified[label_cols].fillna(value=-999) 
    data_modified.to_csv('dataset/data_train_multi.csv', index=False)

    data_modified['stratify_label'] = data_modified[label_cols].astype(str).agg('.'.join, axis=1)
    labels = data_modified['stratify_label']
    groups = data_modified['uniprot_id']
    
    if os.path.exists(folders['splits']['splits']):
        shutil.rmtree(folders['splits']['splits'])
    os.makedirs(folders['splits']['splits'])
    
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(data_modified, labels, groups)):
        train_df = list(data_modified.iloc[train_idx]['identifier'])
        val_df = list(data_modified.iloc[val_idx]['identifier'])
        val_df += [''] * (len(train_df) - len(val_df))
        
        split_df = pd.DataFrame({'train': train_df, 'val': val_df})
        split_df.to_csv(f'{os.path.join(folders["splits"]["splits"], str(fold))}.csv', index=False)
        
        if os.path.exists(f'{os.path.join(folders["splits"]["splits"], str(fold))}.csv'):
            print(f'File number {fold} created!')

if __name__ == '__main__':
    create_split()
