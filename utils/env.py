import os
import yaml
import pandas as pd


def get_from_env(file_path):
    if not os.path.exists(file_path):
            raise FileNotFoundError(f'File does not exist: {file_path}')
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as error:
        print(f'There was a problem --> {error}')
        raise


env_variable = get_from_env('config/config.yaml')
folders = env_variable['folders']

data = pd.read_csv(env_variable['dataset_path'], sep='\t')
