import torch
import platform
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

def get_device():
    """
    Determines the most appropriate device for torch computations based on the available hardware.
    
    This function checks the system's platform and available hardware accelerators (GPU/MPS),
    preferring GPU on non-Mac systems and MPS (Apple's Metal Performance Shaders) on Mac systems 
    when available. If neither is available, it defaults to CPU.

    Returns:
    - device (torch.device): The torch device object indicating the selected hardware device.
    """
    if platform.system().lower() == 'darwin':
        use_gpu = torch.backends.mps.is_built()
        dev_name = "mps"
    elif torch.cuda.is_available():    
        dev_name = "cuda"
    else:
        dev_name = "cpu"
    device = torch.device(dev_name)
    return device

def get_cifar10_loaders(train_transforms, test_transforms, root = './data', shuffle=True, 
                        batch_size=512, pin_memory=True, num_workers=0):
    train_dset = datasets.CIFAR10(root=root, download=True, train=True, transform=train_transforms)
    test_dset = datasets.CIFAR10(root=root, download=True, train=False, transform=test_transforms)    
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    train_loader = torch.utils.data.DataLoader(train_dset,**dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dset,**dataloader_args)
    return train_loader, test_loader