import os
import gzip
import shutil
import pandas as pd

from tqdm import tqdm

source_dir = "dataset/pdb_full"
destination_dir = "ASCARIS/input_files/alphafold_structures"
data_file = "dataset/data_train.csv"
output_txt = "ASCARIS/input_files/mutations.txt"


if __name__ == "__main__":
    os.makedirs(destination_dir, exist_ok=True)

    if len(os.listdir(destination_dir)) == 0:
        for filename in tqdm(os.listdir(source_dir), desc="Compressing and copying files"):
            if filename.endswith(".pdb"):
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, f"{filename}.gz")

                with open(source_path, 'rb') as f_in:
                    with gzip.open(destination_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        print("Compressione e copia completate!")


    if not os.path.exists(output_txt):
        data = pd.read_csv(data_file)
        columns_to_extract = ["uniprot_id", "wildtype", "position", "mutation"]

        if not all(col in data.columns for col in columns_to_extract):
            raise ValueError("Una o più colonne richieste non sono presenti nel file CSV.")

        with open(output_txt, "w") as txt_file:
            for _, row in data.iterrows():
                mutations = row["mutation"].split(",") if pd.notna(row["mutation"]) else []
                for mutation in mutations:
                    line = f"{row['uniprot_id']}\t{row['wildtype']}\t{row['position']}\t{mutation.strip()}\n"
                    txt_file.write(line)

        print("File TXT generato con successo!")
