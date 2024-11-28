import torch.nn as nn
import torch 
from loss import mse_loss
from datasets import SpectrogramDataset

class RNN(nn.Module):
    def __init__(self, mode, input_size, hidden_size, num_layers, dropout, output_size):
        super(RNN, self).__init__()
        
        # Save the mode for use in forward
        self.mode = mode
        
        # Define the appropriate RNN type based on the mode
        if mode == "RNN": 
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                nonlinearity='relu',
                bias=True,
                batch_first=False
            )
        elif mode == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bias=True,
                batch_first=False,
                bidirectional=False
            )
        elif mode == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=False
            )
        else:
            raise ValueError("Undefined mode, enter 'RNN', 'GRU' or 'LSTM'")
        
        # BatchNorm on the hidden size
        self.bn = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        input_size = feature_dim
        
        # Reshape input and permute dimensions
        x = x.view(batch_size, seq_len, input_size)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_size]
        
        # Forward pass through the appropriate RNN type
        if self.mode in ["RNN", "GRU", "LSTM"]:
            out, _ = self.rnn(x)
        else:
            raise ValueError("Undefined mode, enter 'RNN', 'GRU' or 'LSTM'")
        
        # Take the output of the last time step
        out = self.bn(out[-1])  # [batch_size, hidden_size]
        
        # Pass through fully connected layers
        out = self.fc(out)
        return out

   



# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/n**.5
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)
        
def weights_init_kaiming(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
