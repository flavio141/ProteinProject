import os
import torch
import numpy as np
import pandas as pd


def write_information(path, model, optimizer, criterion, args):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Args --> {args}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write("\n")


def save_metrics_to_csv(metrics, epoch, parameters, trial_id, cv):

    metrics["Epoch"] = epoch
    metrics["Trial"] = trial_id
    metrics["CV"] = cv

    df_row = pd.DataFrame([metrics])

    file_path = os.path.join('optuna', parameters['model'], f'trial_{trial_id}_metrics.csv')

    if not os.path.exists(file_path):
        df_row.to_csv(file_path, index=False)
    else:
        df_row.to_csv(file_path, mode='a', index=False, header=False)


def seed_torch(device, seed=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True