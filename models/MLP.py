import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout, out_dim):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.dropout = dropout
        self.layers = []

        for i in range(self.num_layers):
            if i == 0:
                dim1 = input_dim
            else:
                dim1 = hidden_dims[i-1]
            if i == self.num_layers - 1:
                dim2 = out_dim
            else:
                dim2 = hidden_dims[i]
            self.layers.append(nn.Linear(in_features=dim1, out_features=dim2))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return torch.softmax(x, dim=-1)
