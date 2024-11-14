import torch.nn as nn
import torch 

from loss import mse_loss
from datasets import SpectrogramDataset
    

class CNN_97(nn.Module):
    #loss_fn = mse_loss
    loss_fn = nn.MSELoss()
    dataset = SpectrogramDataset

    def __init__(self, 
                 num_conv_layers=4,
                 num_fc_layers=3,
                 conv_dropout=0.2, 
                 linear_dropout=0.1, 
                 kernel_size=5,
                 activation=nn.ReLU,
                 hidden_units=1024,
                 input_shape=(6, 74, 918)):
        super().__init__()
        
        # Define dynamic convolutional layers
        conv_layers = []
        in_channels = input_shape[0]  # Assuming the input has 6 channels
        height, width = input_shape[1], input_shape[2]
        out_channels = 16  # Starting output channels for the first conv layer
        stride = 1

        for _ in range(num_conv_layers):
            conv_layers.append(nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=1,
                                         padding=kernel_size // 2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(activation())
            conv_layers.append(nn.MaxPool2d(kernel_size=2))
            conv_layers.append(nn.Dropout(conv_dropout))
            in_channels = out_channels  # Update input channels for the next layer
            out_channels *= 2  # Double output channels for each subsequent layer
            
            # this is just to track the sizes 
            # Update dimensions after convolution and pooling
            height = (height + 2 * (kernel_size // 2) - kernel_size) // stride + 1  # Convolution output height
            width = (width + 2 * (kernel_size // 2) - kernel_size) // stride + 1    # Convolution output width
            height, width = height // 2, width // 2  # After max pooling with kernel_size=2

        self.conv_layers = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        # Calculate the flattened dimension after convolution layers
        input_dim = height*width*in_channels

        # Define dynamic fully connected layers
        fc_layers = []
        for i in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(input_dim, hidden_units))
            fc_layers.append(activation())
            fc_layers.append(nn.Dropout(linear_dropout))
            input_dim = hidden_units  # Update for the next layer
        
        # Final output layer with a single output feature
        fc_layers.append(nn.Linear(input_dim, 1))
        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, input_data):
        x = self.conv_layers(input_data)
        #print(f"Shape after conv layers (before flatten): {x.shape}")  # Debug shape
        x = self.flatten(x)
        #print(f"Flattened size before fully connected layers: {x.shape}")  # Debug shape after flattening
        return self.fc_layers(x)



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
