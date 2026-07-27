import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LSTMClassifier(nn.Module):
    def __init__(self, n_features=11, hidden=64,layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features,hidden,layers,batch_first=True)
        self.head= nn.Linear(hidden,1)

    def forward(self,x):
        out,_ = self.lstm(x)
        last= out[:,-1,:]
        return self.head(last)

