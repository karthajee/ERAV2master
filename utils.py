import torch
import platform
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
import albumentations as A
import numpy as np
from collections import defaultdict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    
    """
    Creates and returns data loaders for the CIFAR-10 training and test datasets.

    Parameters:
        train_transforms (transforms.Compose): Transformations to apply to the training images.
        test_transforms (transforms.Compose): Transformations to apply to the test images.
        root (str, optional): Root directory where the CIFAR-10 dataset is stored or will be downloaded.
                              Defaults to './data'.
        shuffle (bool, optional): Whether to shuffle the dataset before passing it to the loader.
                                  Defaults to True.
        batch_size (int, optional): Number of images per batch. Defaults to 512.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory
                                     before returning them. Defaults to True.
        num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data
                                     will be loaded in the main process. Defaults to 0.

    Returns:
        tuple: A tuple containing two torch.utils.data.DataLoader objects:
               - train_loader: DataLoader for the training data.
               - test_loader: DataLoader for the testing data.
    """

    train_dset = datasets.CIFAR10(root=root, download=True, train=True, transform=train_transforms)
    test_dset = datasets.CIFAR10(root=root, download=True, train=False, transform=test_transforms)    
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    train_loader = torch.utils.data.DataLoader(train_dset,**dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dset,**dataloader_args)
    return train_loader, test_loader

def get_inv_transform_cifar(img):

    means = (0.49139968, 0.48215827 ,0.44653124)
    stds = (0.24703233, 0.24348505, 0.26158768)
    transform = A.Normalize(mean=[-m/s for m, s in zip(means, stds)], 
                            std=[1/s for s in stds], 
                            always_apply=True,
                            max_pixel_value=1.0)
    return transform(image=np.array(img))['image']

def get_misclassified_images(model, dataloader, classes, device, plot_flag=True, nrows=2):
    model.eval()
    misclassified = defaultdict(list)
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            probs = F.softmax(output, dim=-1)
            y_pred_prob, y_pred = torch.max(probs, dim=-1)
            mis_idxs = (y_pred != labels.view_as(y_pred)).nonzero(as_tuple=True)[0]
            for idx in mis_idxs:
                if len(misclassified['img_numpy']) >= nrows * 5:
                    break                                            
                misclassified['img_numpy'].append(data[idx].cpu().numpy())
                misclassified['y_true'].append(labels[idx].cpu().numpy())
                misclassified['y_pred'].append(y_pred[idx].cpu().numpy())
                misclassified['prob'].append(y_pred_prob[idx].cpu().numpy())
    if plot_flag:
        fig, axs = plt.subplots(nrows, 5, figsize=(15, nrows * 4))
        for i, img in enumerate(misclassified['img_numpy']):
            ax = axs[i // 5, i % 5]
            img = img.transpose(1, 2, 0)
            img_unnorm = get_inv_transform_cifar(img)
            ax.imshow(img_unnorm)
            title = f'True: {classes[misclassified["y_true"][i]]}\nPredicted: {classes[misclassified["y_pred"][i]]}\nConfidence: {misclassified["prob"][i] * 100:.2f}%'
            ax.set_title(title)
            ax.axis('off')
        fig.show()
    return misclassified

def display_grad_cam_batch(model, target_layers, images_dict, device, classes, 
                           gradcam_transparency=0.5, targets=None):

    cam = GradCAM(model, target_layers)
    input_tensor = torch.tensor(np.array(images_dict['img_numpy']), device=device)
    grayscale_cam = cam(input_tensor, targets=targets)        

    num_items = len(images_dict['img_numpy'])
    rows = (num_items-1)//5 + 1
    cols = 5 if num_items >= 5 else num_items
    fig, axs = plt.subplots(nrows=rows, 
                            ncols=cols, 
                            figsize=(3 * cols, 4 * rows))
    
    for idx, ax in enumerate(axs.flat):

        img = images_dict['img_numpy'][idx]
        y_true = images_dict['y_true'][idx]
        y_pred = images_dict['y_pred'][idx]
        conf = images_dict['prob'][idx]
        gradcam_img = grayscale_cam[idx]

        img_unnorm = get_inv_transform_cifar(img.transpose(1, 2, 0))
        viz = show_cam_on_image(img_unnorm, gradcam_img, use_rgb=True, image_weight=gradcam_transparency)

        ax.imshow(viz)
        title = f'True: {classes[images_dict["y_true"][idx]]}\nPredicted: {classes[images_dict["y_pred"][idx]]}\nConfidence: {images_dict["prob"][idx] * 100:.2f}%'
        ax.set_title(title)
        ax.axis('off')
    
    fig.show()