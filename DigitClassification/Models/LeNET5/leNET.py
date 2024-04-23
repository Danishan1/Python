from torch import nn
from time import time
from helperFunction import printTime 


class leNET5(nn.Module):
    def __init__(self, outputLayer: int):
        startTime = time() # Start time of the Model
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1
            ),  # output Shape (6, 28, 28)
            nn.Tanh(),  # No change in shape
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output Shape = (6, 14, 14)
        )

        self.bloack2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=1
            ),  # Output Shape (16, 10, 10)
            nn.Tanh(),  # No change in shape
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output Shape = (16, 4, 4)
        )

        self.block3 = nn.Sequential(
            nn.Flatten(),  # Layer have 16*5*5 Neurons
            nn.Linear(16 * 4 * 4, 120),  # Layer have 120 Neurons
            nn.Linear(120, 84),  # Layer have 84 Neurons
            nn.Linear(84, outputLayer),  # Layer have 10 Neurons
            nn.Softmax(dim=1)
        )
        
        printTime(startTime, "LeNET-5 Model Instance Created Sucessfully")
        

    def forward(self, x):
        x = self.block1(x)
        x = self.bloack2(x)
        x = self.block3(x)
        return x
