import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import os
import time

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # (3, 32, 32) —> (64, 32, 32)
        self.l1 = self.ResLayer(64, 128) # (64, 32, 32) —> (128, 16, 16)
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # (128, 16, 16) —> (256, 8, 8)
        self.l3 = self.ResLayer(256, 512) # (256, 8, 8) —> (512, 4, 4)
        self.o = nn.Sequential(
            nn.MaxPool2d(4), # (512, 4, 4) —> (512, 1, 1)
            nn.Flatten(), # (512, 1, 1) —> (512, )
            nn.Linear(512, 10),            
        )
    
    def forward(self, x):
        x = self.prep(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.o(x)
        return x
    
    def summarise(self, device, in_size=(1, 3, 32, 32)):
        print(summary(self, in_size, device=device))
    
    def save(self, dir='weights/'):
        """
        Saves the current model state with a timestamp and channel configuration in the filename.

        Parameters:
        - dir (str, optional): Directory path where the model will be saved. Defaults to 'weights/'.
        """
        os.makedirs(dir, exist_ok=True)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filepath = f"{dir}/ass10_{timestr}.pt"
        torch.save(self, filepath)
    
    class ResLayer(nn.Module):
        
        def __init__(self, in_c, out_c):
            super().__init__()
            self.con = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding='same'),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
            self.res = self.ResBlock(out_c)
        def forward(self, x):
            x = self.con(x)
            R = self.res(x)
            x = x + R
            return x
    
        class ResBlock(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.residual = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding="same"),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding="same"),
                    nn.BatchNorm2d(c),                    
                )
            
            def forward(self, x):
                x = F.relu(x + self.residual(x))
                return x