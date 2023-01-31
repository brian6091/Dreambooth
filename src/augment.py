import hashlib
import math
import random
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image, ImageStat
from torchvision import transforms

import numpy as np

class pad_to_square(object):
    def __init__(self, fill):
        self.fill = fill

    def __call__(self, image):
        old_size = image.size
        max_dimension, min_dimension = max(old_size), min(old_size)
        desired_size = (max_dimension, max_dimension)
        position = int(max_dimension/2) - int(min_dimension/2)

        if self.fill=="gaussian":
            fill = np.random.normal(0, 3, (max_dimension, max_dimension, 3))
            padded_image = Image.fromarray(fill.astype('uint8'), 'RGB')
        else:
            padded_image = Image.new("RGB", desired_size, color='black')
        
        if image.height<image.width:
            padded_image.paste(image, (0, position))
        else:
            padded_image.paste(image, (position, 0))

        return padded_image

      
class Augmentor(object):
    def __init__(
        self,
        output_dir=None,
        pad_to_square: bool = False,
        min_resolution: int = 512,
        out_resolution: int = 512,
        center_crop: bool = False,
        horizontal_flip: bool = False,
        trivialwide: bool = False,
    ):
        self.output_dir = output_dir
        self.pad_to_square = pad_to_square
        self.min_resolution = min_resolution
        self.out_resolution = out_resolution
        self.center_crop = center_crop
        self.horizontal_flip = horizontal_flip
        self.trivialwide = trivialwide
        
        self.build()

    def build(self):
        augment_list = []
        
        if self.pad_to_square:
            augment_list.append(pad_to_square("gaussian"))
            
        if self.min_resolution is not None:
            augment_list.append(transforms.Resize(self.min_resolution, interpolation=transforms.InterpolationMode.BILINEAR))
            
        if self.center_crop:
            augment_list.append(transforms.CenterCrop(self.out_resolution))
        else:
            augment_list.append(transforms.RandomCrop(self.out_resolution))
            
        if self.horizontal_flip:
            augment_list.append(transforms.RandomHorizontalFlip(0.5))
            
        if self.trivialwide:
            augment_list.append(transforms.TrivialAugmentWide())

        if len(augment_list)>0:
            self.augment = transforms.Compose(augment_list)
        else:
            self.augment = None
            
    def __call__(self, image):
        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        if self.augment is not None:
            image = self.augment(image)
            if self.output_dir is not None:
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = image_path.stem + f"-{hash_image}.jpg"
                image.save(os.path.join(self.augment_output_dir, image_filename))
                
        return image
