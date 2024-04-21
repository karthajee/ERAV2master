import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AssignmentTenTrainTransform:
    """
    A wrapper class for albumentations transformations to make them compatible with PyTorch datasets.
    
    This class ensures that the transformations defined using the albumentations library can be
    directly used in PyTorch datasets by overriding the `__call__` method to process images.

    Parameters:
    - transforms (A.Compose): An albumentations.Compose object encapsulating the desired transformations.

    Methods:
    - __call__(img): Applies the transformations to the input image.
    """

    def __init__(self):
        """
        Initializes the Transforms object with the specified albumentations.Compose transformations.
        """
        self.means = (0.49139968, 0.48215827 ,0.44653124)
        self.stds = (0.24703233, 0.24348505, 0.26158768)
        self.transform = A.Compose(
            [
                A.Normalize(mean=self.means, std=self.stds, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(width=32, height=32, always_apply=True),
                A.HorizontalFlip(),                
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1,
                                min_height=8, min_width=8, fill_value=self.means, mask_fill_value=None),
                ToTensorV2()
            ]
            )        

    def __call__(self, img):
        """
        Applies the predefined transformations to an input image when the object is called.

        Parameters:
        - img (PIL.Image or numpy.ndarray): The input image to transform.

        Returns:
        - numpy.ndarray: The transformed image as a numpy array.
        """
        return self.transform(image=np.array(img))['image']
    
class AssignmentTenTestTransform:
    """
    A wrapper class for albumentations transformations to make them compatible with PyTorch datasets.
    
    This class ensures that the transformations defined using the albumentations library can be
    directly used in PyTorch datasets by overriding the `__call__` method to process images.

    Parameters:
    - transforms (A.Compose): An albumentations.Compose object encapsulating the desired transformations.

    Methods:
    - __call__(img): Applies the transformations to the input image.
    """

    def __init__(self):
        """
        Initializes the Transforms object with the specified albumentations.Compose transformations.
        """
        self.means = (0.49139968, 0.48215827 ,0.44653124)
        self.stds = (0.24703233, 0.24348505, 0.26158768)        
        self.transform = A.Compose([
            A.Normalize(mean=self.means, std=self.stds, always_apply=True),
            ToTensorV2(),
        ])

    def __call__(self, img):
        """
        Applies the predefined transformations to an input image when the object is called.

        Parameters:
        - img (PIL.Image or numpy.ndarray): The input image to transform.

        Returns:
        - numpy.ndarray: The transformed image as a numpy array.
        """
        return self.transform(image=np.array(img))['image']
