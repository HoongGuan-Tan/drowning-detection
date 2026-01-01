import torch.nn as nn
import torch.nn.functional as F
from kan_convolutional.KANConv import KAN_Convolutional_Layer
from kan_convolutional.KANLinear import KANLinear

# Classification Model Class definitions
class KKAN_RGB(nn.Module):
    def __init__(self, grid_size: int = 5):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(in_channels=3,
            out_channels= 6,
            kernel_size= (5,5),
            grid_size = grid_size,
            padding =(0,0)
        )

        self.conv2 = KAN_Convolutional_Layer(in_channels=6,
            out_channels= 16,
            kernel_size = (5,5),
            grid_size = grid_size,
            padding =(0,0)
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 

        self.kan1 = KANLinear(
            400,
            3,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.01,
            scale_base=1,
            scale_spline=1,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[0,1],
        )
        self.name = f"KKAN (RGB) (gs = {grid_size})"

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.flat(x)
        x = self.kan1(x)

        return x

class CNN_RGB(nn.Module):
    def __init__(self):
        super(CNN_RGB, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 3)  # Flattened size = 400, output = 3

        # Activation
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.pool(F.silu(self.conv1(x)))  
        x = self.pool(F.silu(self.conv2(x)))  
        x = self.flat(x)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)  

        return x
    