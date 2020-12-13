import torch, math
from torch import nn
import torch.nn.functional as F
from modules.Attention import AttentionLayer

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout, out_dim, input_features=2, gpu=False):
        super(Attention, self).__init__()
        self.hidden_dims = hidden_dim
        self.num_layers = layers
        self.dropout = dropout
        self.gpu = gpu
        self.layers = []

        self.embedding = nn.Embedding(input_features, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        for i in range(self.num_layers):
            self.layers.append(AttentionLayer(hidden_dim=hidden_dim))
        self.o = nn.Linear(hidden_dim, out_dim)
        self.layers = nn.ModuleList(self.layers)
        self.make_positional_encoding()

    def make_positional_encoding(self, input_dim1=6, input_dim2=7):
        # x [b_size, 42, emb_size]
        encoding1_len = int(math.log(input_dim1, 2) + 1)
        encoding2_len = int(math.log(input_dim2, 2) + 1)
        self.positional_encoding = torch.zeros(input_dim1*input_dim2, self.hidden_dims).long()
        for i in range(input_dim1):
            for ii, e in enumerate(bin(i)[2:].zfill(encoding1_len)):
                self.positional_encoding[[x*input_dim1+i for x in range(input_dim2)], ii] = int(e)
        for i in range(input_dim2):
            for ii, e in enumerate(bin(i)[2:].zfill(encoding2_len)):
                self.positional_encoding[[x*input_dim2+i for x in range(input_dim1)], ii+encoding1_len] = int(e)
        self.positional_encoding = torch.cat([self.positional_encoding, self.positional_encoding], dim=0)
        if self.gpu:
            self.positional_encoding = self.positional_encoding.cuda()

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x += self.positional_encoding
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.layer_norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.max(x, dim=-2).values
        x = self.o(x)
        return torch.softmax(x, dim=-1)
