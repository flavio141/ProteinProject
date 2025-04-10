import os
import esm
import torch
import argparse
import numpy as np

import sys
sys.path.append('utils')

from Bio import SeqIO
from tqdm import tqdm
from env import folders

parser = argparse.ArgumentParser(description='Extract Features')
parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')


def extract_features(sequence_wt, sequence_mt, protein_mt, model, batch_converter, device):
    if len(sequence_wt) > 1022:
        with open("long_sequences.txt", "a") as f:
            f.write(protein_mt + '\t' + str(len(sequence_wt)) + "\n")
        
        device = torch.device('cpu')
        model.to(device)

    _, _, batch_tokens = batch_converter([("wt", sequence_wt), ("mt", sequence_mt)])
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_embeddings = results["representations"][33].cpu().numpy()
    del batch_tokens, results
    
    torch.cuda.empty_cache()


    difference = (token_embeddings[0] - token_embeddings[1])
    np.savez_compressed(f'{os.path.join(folders["embedding"]["wt"], protein_mt)}.npz', token_embeddings[0])
    np.savez_compressed(f'{os.path.join(folders["embedding"]["mut"], protein_mt)}.npz', token_embeddings[1])
    np.savez_compressed(f'{os.path.join(folders["embedding"]["diff"], protein_mt)}.npz', difference)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)


def esm_features(model, batch_converter, device):
    fasta_path = folders['dataset']['fasta']
    fasta_mut_path = folders['dataset']['fasta_mut']

    for filename in tqdm(os.listdir(fasta_path), desc='Extracting features with ESM'):
        if filename.endswith(".fasta"):
            protein_name_wt = filename.split(".")[0]

            seq_record_wt = SeqIO.read(os.path.join(fasta_path, filename), "fasta")
            fasta_present = [fasta_mut for fasta_mut in os.listdir(fasta_mut_path) if protein_name_wt in fasta_mut]

            for fasta_mut in fasta_present:
                protein_name_mut = fasta_mut.split(".")[0]
                if os.path.exists(f'{os.path.join(folders["embedding"]["diff"], protein_name_mut)}.npz'):
                    continue

                seq_record_mut = SeqIO.read(os.path.join(fasta_mut_path, fasta_mut), "fasta")
                extract_features(str(seq_record_wt.seq), str(seq_record_mut.seq), protein_name_mut, model, batch_converter, device)


if __name__ == "__main__":
    args = parser.parse_args()

    for subfolders in folders.values():
        for subfolder in subfolders.values():
            os.makedirs(subfolder, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model.to(device)
    model.eval()

    esm_features(model, batch_converter, device)