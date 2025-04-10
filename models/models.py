import os
import torch
import numpy as np
import torch.nn as nn
import flax.linen as fx
import torch.nn.functional as F

from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    def __init__(self, ids, data, wt_folder, mut_folder, diff_folder, outcome):
        self.data = data
        self.ids = ids
        self.wt_folder = wt_folder
        self.mut_folder = mut_folder
        self.diff_folder = diff_folder

        self.outcome = outcome

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        outcome = self.data.loc[self.ids[idx]][self.outcome]
        wt_feat = torch.tensor(np.load(os.path.join(self.wt_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)
        mut_feat = torch.tensor(np.load(os.path.join(self.mut_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)
        diff_feat = torch.tensor(np.load(os.path.join(self.diff_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)

        return wt_feat, mut_feat, diff_feat, outcome


class ProteinAttDataset(Dataset):
    def __init__(self, ids, data, wt_folder, diff_folder, outcome):
        self.data = data
        self.ids = ids
        self.wt_folder = wt_folder
        self.diff_folder = diff_folder

        self.outcome = outcome

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        outcome = self.data.loc[self.ids[idx]][self.outcome]
        wt_feat = torch.tensor(np.load(os.path.join(self.wt_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)
        diff_feat = torch.tensor(np.load(os.path.join(self.diff_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)

        return wt_feat, diff_feat, outcome
    

class ProteinMultiDataset(Dataset):
    def __init__(self, ids, data, wt_folder, diff_folder, outcome):
        self.data = data
        self.ids = ids
        self.wt_folder = wt_folder
        self.diff_folder = diff_folder

        self.outcome = outcome

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        outcome = self.data.loc[self.ids[idx]][self.outcome]
        wt_feat = torch.tensor(np.load(os.path.join(self.wt_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)
        diff_feat = torch.tensor(np.load(os.path.join(self.diff_folder, self.ids[idx] + ".npz"))['arr_0'], dtype=torch.float32)

        return wt_feat, diff_feat, outcome
    

class ProteinCrossAttentionModel(nn.Module):
    def __init__(self, input_dim=1280, num_heads=2, dropout=0.3):
        super(ProteinCrossAttentionModel, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 15)

    def forward(self, x):
        wt = x[:, 0, :, :]
        diff = x[:, 1, :, :]
        
        x, _ = self.cross_attention(query=diff, key=wt, value=wt, need_weights=False)
        
        x = x.transpose(1, 2)
        x = self.pooling(x).squeeze(-1)
        
        x = F.gelu(self.fc1(x))
        x = self.dropout1(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

    

class ProteinJax(fx.Module):
    input_dim: int = 1280
    num_heads: int = 4

    def setup(self):
        self.cross_attention = fx.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.fc = fx.Dense(3)

    def __call__(self, x):
        wt = x[:, 0, :, :]
        diff = x[:, 1, :, :]
        
        attn_output = self.cross_attention(diff, wt, wt)
        output = self.fc(attn_output.mean(axis=1))
        return output



class CrossAttentionExpert(nn.Module):
    def __init__(self, input_dim=1280, num_heads=8):
        super(CrossAttentionExpert, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, wt, x):
        attn_output, _ = self.cross_attention(query=wt, key=x, value=x, need_weights=False)
        return self.fc(attn_output.mean(dim=1))


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=1280, num_heads=2, num_classes=3, num_experts=3):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([CrossAttentionExpert(input_dim, num_heads) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        wt = x[:, 0, :, :]
        mut = x[:, 1, :, :]
        diff = x[:, 2, :, :]

        inputs = [mut, diff]
        expert_outputs = []

        for i, expert in enumerate(self.experts):
            expert_output = expert(wt, inputs[i % len(inputs)])
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)

        gate_scores = F.softmax(self.gate(wt.mean(dim=1)), dim=1)
        fused_output = torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=1)
        logits = self.classifier(fused_output)

        return logits


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)

class MixtureOfExpertsLight(nn.Module):
    def __init__(self, input_dim, num_classes=3, num_experts=2):
        super(MixtureOfExpertsLight, self).__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([Expert(input_dim, num_classes) for _ in range(num_experts)])
        self.gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        wt, diff = x[:, 0, :, :], x[:, 1, :, :]

        gate_logits = self.gating(wt.mean(dim=1))
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([self.experts[i](x_i.mean(dim=1)) for i, x_i in enumerate([wt, diff])], dim=1)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output
