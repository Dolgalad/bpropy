import torch
import torch.nn.functional as F
from torch.nn import ReLU, ELU, Dropout
from torch_geometric.nn import GCNConv, Sequential

class GCN_v1(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout_p=0.25):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv1 = GCNConv(9, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv4 = GCNConv(hidden_dim, 1, normalize=False)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = x + self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = x + self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv4(x, edge_index)

        return x

class GCN_v2(torch.nn.Module):
    def __init__(self, in_dim, n_layers=4, hidden_dim=64, dropout_p=0.25, normalize=False):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv_in = GCNConv(9, hidden_dim, normalize=normalize)
        self.layers = Sequential("x,edge_index", [
            (Sequential("x,edge_index", [
                (GCNConv(hidden_dim, hidden_dim, normalize=normalize), "x,edge_index -> x"),
                ELU(inplace=True),
                Dropout(dropout_p)
                ]), "x,edge_index -> x") for _ in range(n_layers)
            ])
        self.conv_out = GCNConv(hidden_dim, 1, normalize=normalize)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        x = self.conv_in(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.layers(x, edge_index)

        x = self.conv_out(x, edge_index)

        return x

