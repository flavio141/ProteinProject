import os
import torch
import numpy as np
import pandas as pd


def write_information(args, path, model, optimizer, criterion, epochs, scheduler=None):
    with open(path, "w") as file:
        file.write("Model Information\n\n")
        file.write(f"Epochs --> {epochs}\n")
        file.write(f"Model Architecture --> {model}\n")
        file.write(f"Optimizer --> {optimizer}\n")
        file.write(f"Loss Function --> {criterion}\n")
        file.write(f"Scheduler --> {scheduler}\n")
        file.write("\n")
        file.write("Argoments Information\n\n")
        file.write(f"Args --> {args}\n")
        file.write("\n")


def save_metrics_to_csv(metrics, epoch, file_path='metrics.csv'):
    if not isinstance(metrics, dict):
        raise ValueError("Metrics should be a dictionary with metric names as keys and their values.")

    metrics["Epoch"] = epoch

    df_row = pd.DataFrame([metrics])

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