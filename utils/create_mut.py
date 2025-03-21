import os
from tqdm import tqdm

from env import folders


def create_fasta_mutated(wildtype, position, mutation, pdb):
    if len(os.listdir(folders['dataset']['fasta_mut'])) == len(pdb.unique()):
        print('All FASTA mutated already created')
        return
    elif len(os.listdir(folders['dataset']['fasta'])) == 0:
        print('There are no FASTA files to mutate')
        return

    try:
        for wt, pos, mut, pdb, _ in zip(wildtype, position, mutation, pdb, tqdm(range(0, len(pdb)), desc= 'Extracting FASTA Mutated')):

            with open(f"{os.path.join(folders['dataset']['fasta'], pdb)}.fasta", 'r') as fasta:
                fasta_original = fasta.read()

            pos_alpha = pos - 1
            
            if len(fasta_original.split('\n')[1]) >= pos_alpha and fasta_original.split('\n')[1][pos_alpha] == wt:
                if ',' in mut:
                    for m in mut.split(','):
                        if f'{pdb}_{wt}_{pos}_{m}.fasta' in os.listdir(folders['dataset']['fasta_mut']):
                            continue
                        fasta_seq = fasta_original.split('\n')[1][:pos_alpha] + m + fasta_original.split('\n')[1][pos_alpha + 1:]
                        fasta_mut = fasta_original.split('\n')[0] + '\n' + fasta_seq

                        with open(f"{os.path.join(folders['dataset']['fasta_mut'], pdb)}_{wt}_{pos}_{m}.fasta", 'w') as mutated:
                            mutated.write(fasta_mut)
                else:
                    if f'{pdb}_{wt}_{pos}_{mut}.fasta' in os.listdir(folders['dataset']['fasta_mut']):
                        continue
                    fasta_seq = fasta_original.split('\n')[1][:pos_alpha] + mut + fasta_original.split('\n')[1][pos_alpha + 1:]
                    fasta_mut = fasta_original.split('\n')[0] + '\n' + fasta_seq

                    with open(f"{os.path.join(folders['dataset']['fasta_mut'], pdb)}_{wt}_{pos}_{mut}.fasta", 'w') as mutated:
                        mutated.write(fasta_mut)
            elif len(fasta_original.split('\n')[1]) >= pos_alpha:
                original = fasta_original.split("\n")[1][pos_alpha]
                print(f'No match between {wt} and the amino acids {original} at position:{pos} for {pdb}')
            else:
                length = len(fasta_original.split('\n')[1])
                print(f"The position is out of range: {pos} and length is {length} for {pdb}")
    except Exception as error:
        print(f'There was an error: {error}')
        assert False