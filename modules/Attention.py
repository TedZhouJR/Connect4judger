import torch
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.Q_trans = nn.Linear(hidden_dim, hidden_dim)
        self.K_trans = nn.Linear(hidden_dim, hidden_dim)
        self.V_trans = nn.Linear(hidden_dim, hidden_dim)
        self.O_trans = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        q = self.Q_trans(x)
        k = self.K_trans(x)
        v = self.V_trans(x)
        k = k.transpose(2, 1)
        logits = torch.matmul(q, k)
        logits = nn.functional.softmax(logits, dim=-1)
        y = torch.matmul(logits, v)
        y = self.O_trans(y)
        return y