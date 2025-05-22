import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BinarySVM(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()



class BinarySVM_(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.feature_extractor(x).squeeze()