import os
import argparse

import sys
sys.path.append('utils')

from create_mut import create_fasta_mutated
from download import download_pdb, extract_fasta, data, folders


parser = argparse.ArgumentParser(description=('Extract PDB files & FASTA sequences'))
parser.add_argument('--extract_pdb', default=False, help='Tell if necessary to extract PDB files')
parser.add_argument('--extract_fasta_wt', default=False, help='Tell if necessary to extract FASTA sequences for wildtype proteins')
parser.add_argument('--extract_fasta_mut', default=True, help='Tell if necessary to extract FASTA sequences for mutated proteins')
parser.add_argument('--rcsb', default=False, help='Tell if necessary to extract pdb files from RCSB')
args = parser.parse_args()



def extract_pdb_files():
    try:
        nans = {'wildtype' : int(data['wildtype'].isnull().sum()), 
                'position': int(data['position'].isnull().sum()), 
                'mutation': int(data['mutation'].isnull().sum())
            }
        
        if nans['wildtype'] != 0 or nans['position'] != 0 or nans['mutation'] != 0:
            data.dropna(subset = [max(nans.items(), key=lambda x: x[1])[0]])

        download_pdb(data['uniprot_id'].unique(), args)
        
    except Exception as error:
        print(f'There was an error: {error}')
        assert False


def extract_fasta_mut():
    try:
        fasta_ids = [file.split('.')[0] for file in os.listdir(folders['dataset']['fasta'])]
        filtered_data = data[data["uniprot_id"].isin(fasta_ids)]
        create_fasta_mutated(filtered_data['wildtype'], filtered_data['position'], filtered_data['mutation'], filtered_data['uniprot_id'])
    except Exception as error:
        print(f'There was an error: {error}')
        assert False


if __name__== "__main__":
    for subfolders in folders.values():
        for subfolder in subfolders.values():
            os.makedirs(subfolder, exist_ok=True)

    if args.extract_pdb == True:
        extract_pdb_files()

    if args.extract_fasta_wt == True:
        extract_fasta()

    if args.extract_fasta_mut == True:
        extract_fasta_mut()
