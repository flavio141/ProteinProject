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


def extract_around_difference(seq_record_wt, seq_record_mut, flanking=500):
    if flanking > 510:
        raise ValueError("Flanking must be less than 510")

    max_length = flanking * 2 + 1

    for i, (aa_wt, aa_mut) in enumerate(zip(seq_record_wt, seq_record_mut)):
        if aa_wt != aa_mut:
            diff_pos = i
            break
    else:
        raise ValueError("No difference found between the sequences")

    start_pos = max(0, diff_pos - flanking)
    end_pos = min(len(seq_record_wt), diff_pos + flanking + 1)

    length = end_pos - start_pos
    if length < max_length:
        if diff_pos - start_pos < flanking and end_pos - diff_pos == (flanking + 1):
            end_pos = end_pos + flanking - (diff_pos - start_pos)
        else:
            start_pos = start_pos - flanking + (end_pos - diff_pos) - 1

    sub_seq_wt = seq_record_wt[start_pos:end_pos]
    sub_seq_mut = seq_record_mut[start_pos:end_pos]
    if len(sub_seq_wt) != 1001:
        raise ValueError("The sequence length is not 1001")

    return sub_seq_wt, sub_seq_mut, end_pos, start_pos


def extract_features(sequence_wt, sequence_mt, protein_mt, model, batch_converter, device):
    if len(sequence_wt) > 1022:
        sequence_wt, sequence_mt, _, _ = extract_around_difference(sequence_wt, sequence_mt)

    _, _, batch_tokens = batch_converter([("wt", sequence_wt), ("mt", sequence_mt)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])

    token_embeddings = results["representations"][33].cpu().numpy()
    del batch_tokens, results
    
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_embeddings[i, 1 : tokens_len - 1])
    
    torch.cuda.empty_cache()


    difference = (sequence_representations[0] - sequence_representations[1])
    np.savez_compressed(f'{os.path.join(folders["embedding"]["fastaEmb"], protein_mt)}.npz', difference)


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
                if os.path.exists(f'{os.path.join(folders["embedding"]["fastaEmb"], protein_name_mut)}.npz'):
                    continue

                seq_record_mut = SeqIO.read(os.path.join(fasta_mut_path, fasta_mut), "fasta")
                extract_features(str(seq_record_wt.seq), str(seq_record_mut.seq), protein_name_mut, model, batch_converter, device)


if __name__ == "__main__":
    args = parser.parse_args()

    for subfolders in folders.values():
        for subfolder in subfolders.values():
            os.makedirs(subfolder, exist_ok=True)

    if not args.cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()#esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    model.to(device)
    model.eval()

    esm_features(model, batch_converter, device)