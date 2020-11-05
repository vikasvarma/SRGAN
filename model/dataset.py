import torch
import os
import random
from PIL import Image
from torchvision.transforms import functional as F

class SRDataset(torch.utils.data.Dataset):
    """
        Dataset class for sequential patch loading.
    """
    
    def __init__(
        self, 
        path,
        mode = 'train',
        scale_factor = 4
    ):
        """Super Resolution (SR) Dataset"""
        
        super(SRDataset, self).__init__()
        self.root = path
        self.images = os.listdir(self.root)
        self.scale_factor = scale_factor
        self.mode = mode
        self.crop_size = scale_factor * 96
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """
        Return transformed Low Resolution (LR) and corresponding High-Resolution (HR) images.
        """
        
        # Load the high-resolution (HR) image:
        img_file = os.path.join(self.root, self.images[index])
        img = Image.open(img_file)
        lr, hr = self._transform_(img)

        return lr, hr

    def _transform_(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        if self.mode == 'train':
            # For training, random crop the HR image:
            left = random.randint(1, img.width  - self.crop_size)
            top  = random.randint(1, img.height - self.crop_size)
            hr   = img.crop((left,top,left+self.crop_size, top+self.crop_size))

        else:
            # For evaluation, centre crop the HR image instead to get an image 
            # size which is a perfect factor of the scale.
            left   = (img.width  % self.scale_factor) // 2
            top    = (img.height % self.scale_factor) // 2
            right  = left + (img.width  - (img.width  % self.scale_factor))
            bottom = top  + (img.height - (img.height % self.scale_factor))
            hr     = img.crop((left, top, right, bottom))

        # Down-scale to obtain low-resolution image (LR):
        lr = hr.resize(
            (int(hr.width  / self.scale_factor), 
             int(hr.height / self.scale_factor)),
            Image.BICUBIC
        )

        # Convert the LR and HR image to pytorch tensors:
        hr = F.to_tensor(hr)
        lr = F.to_tensor(lr)

        # Normalize the image tensors:
        hr = 2*hr - 1
        lr = 2*lr - 1
        
        return lr, hr