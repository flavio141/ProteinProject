import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAT, GATConv, GCNConv, ChebConv, ClusterPooling, global_mean_pool, global_add_pool

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, heads=1):
        super(GATModel, self).__init__()
        self.conv1 = GAT(in_channels, hidden_channels, num_layers=heads, dropout=dropout, v2=True)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, add_features):
        x, edge_index = graph.features, graph.edge_index
        x = self.conv1(x, edge_index)
        x = global_mean_pool(x, graph.batch)

        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    

class ComplexGATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(ComplexGATModel, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.conv_last = GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)

        # self.pool1 = ClusterPooling(hidden_channels, "log_softmax", dropout=dropout, threshold=None)
        # self.pool_last = ClusterPooling(hidden_channels, "log_softmax", dropout=dropout, threshold=None)

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)

        self.dropout = nn.AlphaDropout(p=dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.features, graph.edge_index, graph.batch

        x = F.selu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.selu(self.conv_last(x, edge_index))
        x = self.dropout(x)

        # x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        # x, edge_index, batch, _ = self.pool_last(x, edge_index, batch)

        x = global_mean_pool(x, batch)

        x = F.selu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x
    

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(GCNModel, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=5, normalization='rw')
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5, normalization='rw')
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.AlphaDropout(p=dropout)

    def forward(self, graph):
        x, edge_index, batch = graph.features, graph.edge_index, graph.batch

        x = F.selu(self.conv1(x, edge_index))
        x = self.dropout(x)

        x = F.selu(self.conv2(x, edge_index))
        x = self.dropout(x)

        x = global_add_pool(x, batch) 

        x = self.fc(x)
        return x
