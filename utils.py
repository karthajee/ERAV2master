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
import multiprocessing
import torch.nn.functional as F
from PIL import Image

def get_avail_cpu_cores():
    num_cores_avail = multiprocessing.cpu_count()-1
    return num_cores_avail


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

    """
    Applies an inverse transformation to a normalized CIFAR-10 image, returning it to its original form.

    Parameters:
        img (PIL.Image or numpy.ndarray): The image to be transformed back to its original state.

    Returns:
        numpy.ndarray: The de-normalized image.
    """

    means = (0.49139968, 0.48215827 ,0.44653124)
    stds = (0.24703233, 0.24348505, 0.26158768)
    transform = A.Normalize(mean=[-m/s for m, s in zip(means, stds)], 
                            std=[1/s for s in stds], 
                            always_apply=True,
                            max_pixel_value=1.0)
    return transform(image=np.array(img))['image']

def get_misclassified_images(model, dataloader, classes, device, plot_flag=True, nrows=2):

    """
    Identifies and optionally plots misclassified images from a dataloader using a trained model.

    Parameters:
        model (torch.nn.Module): The trained model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the dataset to evaluate.
        classes (list): List of class names corresponding to dataset labels.
        device (torch.device): The device on which the model is located (CPU or GPU).
        plot_flag (bool, optional): If True, misclassified images are plotted. Defaults to True.
        nrows (int, optional): Number of rows in the plot of misclassified images. Defaults to 2.

    Returns:
        dict: A dictionary containing arrays of misclassified image data, true labels, predicted labels, and prediction probabilities.
    """

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

    """
    Displays a batch of images with Grad-CAM overlays to indicate areas of the image that influenced the model's predictions.

    Parameters:
        model (torch.nn.Module): The trained model to visualize the Grad-CAM for.
        target_layers (list of torch.nn.modules): List of layers for which to compute the Grad-CAM.
        images_dict (dict): Dictionary containing arrays of image data and associated labels and prediction info.
                            Keys should include 'img_numpy', 'y_true', 'y_pred', and 'prob'.
        device (torch.device): The device (CPU or GPU) where computations will be performed.
        classes (list): List of class names corresponding to dataset labels.
        gradcam_transparency (float, optional): Transparency level of the Grad-CAM overlay on the original image. Defaults to 0.5.
        targets (list of int, optional): Specific classes for which to generate Grad-CAM. If None, the predicted class is used. Defaults to None.
    """

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

def export_cifar_image(img_numpy, dir='./', filename='output.png'):
    
    """
    Saves a CIFAR-10 image from a numpy array to a file, converting it back to its original visual form if necessary.

    Parameters:
        img_numpy (numpy.ndarray): The image data as a numpy array, expected in CHW format (channels, height, width).
        dir (str, optional): Directory where the image file will be saved. Defaults to the current directory.
        filename (str, optional): The name of the file to save the image as. Defaults to 'output.png'.
    """
    
    img_numpy = img_numpy.transpose(1, 2, 0)
    if (img_numpy < 0).any():
        img_numpy = get_inv_transform_cifar(img_numpy)
    img_int = np.uint8(255 * img_numpy)
    img = Image.fromarray(img_int)
    filepath = dir + filename
    img.save(filepath)