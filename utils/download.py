import os
import json
import requests as r

from urllib.request import urlopen
from tqdm import tqdm
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.Data import IUPACData


from env import data, folders


def extract_atom_lines(pdb_content):
    atom_lines = [line for line in pdb_content.splitlines() if line.startswith("ATOM")]
    return f"\n".join(atom_lines) + "\n"


def download_pdb_alphafold(cIDs, args):
    try:
        if len(cIDs) == len(os.listdir(folders['dataset']['pdb_full'])):
            print('All PDB files are already downloaded')
            return

        not_downloaded_path = 'not_downloaded.txt'
        pdb_alternatives = {
            'alphafold': 'https://alphafold.com/api/prediction/',
            'rcsb': 'https://files.rcsb.org/download/'
        }

        for cID, _ in zip(cIDs, tqdm(range(0, len(cIDs)), desc='Extracting PDB using UniProt ID')):
            if f"{cID}.pdb" in os.listdir(folders['dataset']['pdb_full']):
                continue


            pdb_url = f"{pdb_alternatives['alphafold']}{cID}"
            pdb_response = r.get(pdb_url, timeout=20)
            if pdb_response.status_code == 200:
                pdb_path = os.path.join(folders['dataset']['pdb_full'], f"{cID}.pdb")

                pdb_file = json.loads(pdb_response.content.decode('utf8'))[0]['pdbUrl']
                with urlopen(pdb_file) as response:
                    pdb = response.read()
                with open(pdb_path, 'wb') as f:
                    f.write(pdb)
                continue
            

            if args.rcsb:
                for pdb_id in list(data[data['uniprot_id'] == cID]['pdb_id'].unique()):
                    combined_pdb_content = ""

                    pdb_url_rcsb = f"{pdb_alternatives['rcsb']}{pdb_id.split(':')[0]}.pdb"
                    pdb_response = r.get(pdb_url_rcsb, timeout=10)
                    if pdb_response.status_code == 200:
                        pdb_content = pdb_response.text

                        combined_pdb_content = extract_atom_lines(pdb_content)
                
                    pdb_path = os.path.join(folders['dataset']['pdb_chain'], f"{cID}_{pdb_id}.pdb")
                    with open(pdb_path, 'w') as f:
                        f.write(combined_pdb_content) # type: ignore
                
                continue


            with open(not_downloaded_path, 'a') as not_file:
                not_file.write(f"{cID}\n")

    except r.exceptions.RequestException as e:
        print(f'General error: {e}')
        assert False


def three_to_one(residue_name):
    try:
        return IUPACData.protein_letters_3to1[residue_name]
    except KeyError:
        return 'X'


def extract_fasta():
    if len(os.listdir(folders['dataset']['fasta'])) == len(data['uniprot_id'].unique()):
        print('All FASTA files are downloaded')
        return
    
    try:
        for cID, _ in zip(os.listdir(folders['dataset']['pdb_full']), tqdm(range(0, len(os.listdir(folders['dataset']['pdb_full']))), desc= 'Extracting FASTA')):
            pdb_file = os.path.join(folders['dataset']['pdb_full'], cID)

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(cID.split('.')[0], pdb_file)
            model = structure[0] # type: ignore

            seq_records = []
            for chain in model:
                seq = []
                residue_indices = []
                for residue in chain:
                    if PDB.Polypeptide.is_aa(residue, standard=True):
                        residue_indices.append(int(residue.id[1]))
                        seq.append(three_to_one(residue.resname.capitalize()))
                if seq:
                    full_seq = ['X'] * (max(residue_indices))
                    for aa, index in zip(seq, residue_indices):
                        full_seq[index - 1] = aa
                    seq_str = ''.join(full_seq)
                    seq_records.append(seq_str)

            fasta = f'>{cID.split(".")[0]}\n{"".join(seq_records)}'
            output_file = f"{os.path.join(folders['dataset']['fasta'], cID.split('.')[0])}.fasta"
            with open(output_file, 'w') as file:
                file.write(fasta)
    except Exception as error:
        print(f'There was an error: {error}')
        assert False
