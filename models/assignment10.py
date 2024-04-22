import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import os
import time

class Net(nn.Module):

    """
    A convolutional neural network model comprising several layers including 
    convolutional layers, batch normalization, ReLU activations, max pooling, 
    and a fully connected layer for classification. The network uses residual 
    connections in some layers to enhance learning and prevent the vanishing gradient problem.

    Attributes:
        prep (nn.Sequential): A preparatory layer that processes the initial input.
        l1 (ResLayer): The first residual layer.
        l2 (nn.Sequential): A sequential layer that includes convolution, max pooling,
                            batch normalization, and ReLU activation.
        l3 (ResLayer): The second residual layer.
        o (nn.Sequential): The output layer consisting of max pooling, flattening,
                           and a linear transformation to the final output.
    """
    
    def __init__(self):

        """
        Initializes the neural network layers and constructs the architecture.
        """

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

        """
        Defines the forward pass of the model.

        Parameters:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The output of the network after processing the input tensor through
                          various layers defined in __init__.
        """

        x = self.prep(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.o(x)
        return x
    
    def summarise(self, device, in_size=(1, 3, 32, 32)):

        """
        Prints a summary of the model architecture, including the output shape and parameter count of each layer.

        Parameters:
            device (torch.device): The device the model will be summarized on.
            in_size (tuple): The input size of the tensor, in the format (batch_size, channels, height, width).
        """

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
        
        """
        A residual layer that applies convolutional operations followed by a max pooling,
        batch normalization, and ReLU activation, then adds the result of a residual block.

        Attributes:
            con (nn.Sequential): Convolutional operations before the residual connection.
            res (ResBlock): A residual block that modifies the input and adds it back to the original.
        """
        
        def __init__(self, in_c, out_c):

            """
            Initializes the layer
            """

            super().__init__()
            self.con = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding='same'),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            )
            self.res = self.ResBlock(out_c)
        
        def forward(self, x):

            """
            Applies the convolutional operations, then adds the output of the residual block to the result.

            Parameters:
                x (torch.Tensor): Input data tensor.

            Returns:
                torch.Tensor: The output tensor after applying the layer operations and the residual connection.
            """

            x = self.con(x)
            R = self.res(x)
            x = x + R
            return x
    
        class ResBlock(nn.Module):

            """
            A smaller block used within a ResLayer to apply additional convolutional operations
            which are added back to the block's input.

            Attributes:
                residual (nn.Sequential): The set of convolutional and normalization layers.
            """

            def __init__(self, c):

                """
                Initializes the block
                """

                super().__init__()
                self.residual = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding="same"),
                    nn.BatchNorm2d(c),
                    nn.ReLU(),
                    nn.Conv2d(c, c, 3, padding="same"),
                    nn.BatchNorm2d(c),                    
                )
            
            def forward(self, x):

                """
                Applies residual operations and a ReLU activation after adding the input to the output.

                Parameters:
                    x (torch.Tensor): Input data tensor to the residual block.

                Returns:
                    torch.Tensor: Output tensor after adding the residual to the input and applying ReLU.
                """
                
                x = F.relu(x + self.residual(x))
                return x