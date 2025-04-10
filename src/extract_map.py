import os
import torch
import argparse
import numpy as np
import networkx as nx

import sys
sys.path.append('utils')

from env import folders, data

from Bio import SeqIO
from tqdm import tqdm
from Bio.SeqUtils import seq1
from Bio.PDB.PDBParser import PDBParser

parser = argparse.ArgumentParser(description='Extract Contact Maps')
parser.add_argument('--ca_threshold', type=float, default=6.0, help='Threshold for Cα-Cα contact map')
parser.add_argument('--cb_threshold', type=float, default=8.0, help='Threshold for Cβ-Cβ contact map')
parser.add_argument('--save', type=bool, default=True, help='Save the distance maps')


def create_map():
    directory = folders['dataset']['fasta_mut']
    protein_dict = {}

    for filename in os.listdir(directory):
        if filename.endswith('.fasta'):
            file_no_ext = filename.split('.')[0]
            
            protein_id = file_no_ext.split('_')[0]
            
            if protein_id not in protein_dict:
                protein_dict[protein_id] = []
            protein_dict[protein_id].append(file_no_ext)
    return protein_dict


def extract_attributes(sequence, protein_name, position):
    attributes = {}
    features_dict = np.load(os.path.join(folders['embedding']['fastaEmb'], f"{protein_name}.npz"))
    features = features_dict.f.arr_0

    for new_node, (node, a) in enumerate(zip(position, sequence)):
        attributes[new_node] = {'aminoacid_name': a, 'features': torch.from_numpy(features[node,:])}
    return attributes


def get_ca_cb_coordinates(structure, end_pos, start_pos):    
    ca_coords, amino_ca, pos_ca, cb_coords, amino_cb, pos_cb = [], [], [], [], [], []
    model = structure[0] # type: ignore

    for chain in model:
        for node, residue in enumerate(chain):
            if 'CA' in residue and (end_pos >= int(residue.id[1]) >= (start_pos + 1)):
                ca_coords.append(residue['CA'].get_coord())
                amino_ca.append(seq1(residue.get_resname()))
                pos_ca.append(node - start_pos)

            if 'CB' in residue and (end_pos >= int(residue.id[1]) >= (start_pos + 1)):
                cb_coords.append(residue['CB'].get_coord())
                amino_cb.append(seq1(residue.get_resname()))
                pos_cb.append(node - start_pos)

    return np.array(ca_coords), amino_ca, pos_ca, np.array(cb_coords), amino_cb, pos_cb


def calculate_distance_map(coords):
    n = len(coords)
    distance_map = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distance_map[i, j] = np.linalg.norm(coords[i] - coords[j])
    return distance_map


def calculate_contact_map(distance_map, threshold=8.0):
    contact_map = (distance_map < threshold).astype(int)

    G = nx.from_numpy_array(contact_map)
    return G


if __name__ == "__main__":
    args = parser.parse_args()

    for subfolders in folders.values():
        for subfolder in subfolders.values():
            os.makedirs(subfolder, exist_ok=True)

    map_protein = create_map()

    for pdb_file in tqdm(os.listdir(folders['dataset']['pdb_full']), desc='Extracting Contact Maps for Cα and Cβ'):
        protein_name = pdb_file.split('.')[0]
        pdb = os.path.join(folders['dataset']['pdb_full'], pdb_file)

        parser = PDBParser()
        structure = parser.get_structure(protein_name, pdb)

        sequence_wt = str(SeqIO.read(os.path.join(folders['dataset']['fasta'], f"{protein_name}.fasta"), "fasta").seq)
        
        for protein_mut in map_protein[protein_name]:
            if os.path.exists(os.path.join(folders['embedding']['graphs'], f"{protein_mut}.pt")):
                continue
            
            sequence_mut = str(SeqIO.read(os.path.join(folders['dataset']['fasta_mut'], f"{protein_mut}.fasta"), "fasta").seq)

            ca_coords, amino_ca, pos_ca, cb_coords, amino_cb, pos_cb = get_ca_cb_coordinates(structure, len(sequence_wt), 0)

            ca_distance_map = calculate_distance_map(ca_coords)
            cb_distance_map = calculate_distance_map(cb_coords)

            if args.save:
                np.savez_compressed(os.path.join(folders['contact_map']['ca'], f"{protein_mut}_ca.npz"), ca_distance_map)
                np.savez_compressed(os.path.join(folders['contact_map']['cb'], f"{protein_mut}_cb.npz"), cb_distance_map)

            G_ca = calculate_contact_map(ca_distance_map, args.ca_threshold)
            G_cb = calculate_contact_map(cb_distance_map, args.cb_threshold)

            attributes_ca = extract_attributes(amino_ca, protein_mut, pos_ca)
            attributes_cb = extract_attributes(amino_cb, protein_mut, pos_cb)

            nx.set_node_attributes(G_ca, attributes_ca)
            nx.set_node_attributes(G_cb, attributes_cb)

            torch.save([G_ca, G_cb], os.path.join(folders['embedding']['graphs'], f"{protein_mut}.pt"))
