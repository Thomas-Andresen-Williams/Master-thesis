import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features,task = 'classification', dropout = 0.0):
        super().__init__()
        self.ll1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.ll2 = nn.Linear(in_features = hidden_size, out_features = hidden_size // 2)
        self.ll3 = nn.Linear(in_features=hidden_size // 2, out_features=out_features)
        if task == 'classification':
            self.activation = nn.Sigmoid() 
        else:
            # Temp
            self.activation = nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        x = self.ll1(input)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ll3(x)
        x = self.activation(x)
        return x



class LSTM(nn.Module):
    def __init__(
        self,
        lstm_hidden_size=64,
        linear_hidden_size=64,
        num_layers=1,
        num_features: int = 4,
        dropout: float = 0.0,
        output_length: int = 1,
    ):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        # self.hidden_layer = nn.Linear(lstm_hidden_size, linear_hidden_size)
        self.output_layer = nn.Linear(lstm_hidden_size, output_length)

        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        hidden_and_cell_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if hidden_and_cell_state is not None:
            x, (h_n, c_n) = self.lstm(x, hidden_and_cell_state)
        else:
            x, (h_n, c_n) = self.lstm(x)

        self.hidden_state = h_n
        self.cell_state = c_n
        
        # Since the LSTM gives you all the outputs, you must select only the last:
        # x = self.hidden_layer(x[:,:,:])
        # x = F.relu(x)
        x = self.output_layer(x)
        return x
