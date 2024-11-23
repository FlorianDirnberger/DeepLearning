import torch.nn as nn
import torch 
from loss import mse_loss
from datasets import SpectrogramDataset

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, feature_dim1, feature_dim2 = x.shape
        input_size = feature_dim1 * feature_dim2
        x = x.view(batch_size, seq_len, input_size)
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, input_size]

        out, _ = self.lstm(x)
        out = self.bn(out[-1])  # Apply BatchNorm on the last time step
        out = self.fc(out)      # Pass through fully connected layers
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