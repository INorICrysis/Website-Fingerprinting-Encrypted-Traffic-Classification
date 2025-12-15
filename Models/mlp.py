
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=9, num_classes=20,
                 hidden1=128, hidden2=64, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
