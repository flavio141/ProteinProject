import os
import torch
import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class NormalDataset(Dataset):
    def __init__(self, graph_dir, add_features, ids, outcomes):
        self.graph_dir = graph_dir
        self.add_features = add_features
        self.ids = ids
        self.outcomes = outcomes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        graph_id = self.ids[idx]
        outcome = self.outcomes[idx]

        add_features = torch.tensor(self.add_features.loc[graph_id].values[:-1].reshape(-1,1))
        
        graph_path = os.path.join(self.graph_dir, f"{graph_id}.pt")
        graph = torch.load(graph_path)[0]

        data = from_networkx(graph)
        data.y = outcome

        return data, add_features


class GraphDataset(Dataset):
    def __init__(self, graph_dir, ids, outcomes):
        self.graph_dir = graph_dir
        self.ids = ids
        self.outcomes = outcomes

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        graph_id = self.ids[idx]
        outcome = self.outcomes[idx]
        
        graph_path = os.path.join(self.graph_dir, f"{graph_id}.pt")
        graph = torch.load(graph_path)[0]

        data = from_networkx(graph)
        data.y = outcome

        return data
    

class LossWrapper(torch.nn.Module):
	def __init__(self, loss:torch.nn.Module, ignore_index:int):
		super(LossWrapper, self).__init__()
		self.loss = loss
		self.ignore_index = ignore_index
	
	def __call__(self, input, target):
		input = input.view(-1)
		target = target.view(-1)
		if self.ignore_index != None:
			mask = target.ne(self.ignore_index)
			input = torch.masked_select(input, mask)
			target = torch.masked_select(target, mask)
		
		r = self.loss(input, target)
		return r
