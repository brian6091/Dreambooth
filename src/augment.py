#    Copyright 2022 B. Lau, brian6091@gmail.com
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import hashlib
import math
import random
import os
from pathlib import Path

from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, Iterable

import torch
import torch.nn.functional as F

from PIL import Image, ImageStat, ImageFilter 
from torchvision import transforms

import numpy as np

class pad_to_square(object):
    def __init__(
        self,
        fill: str = "gaussian",
        smooth: int = None,
    ):
        self.fill = fill
        self.smooth = smooth

    def __call__(self, image):
        old_size = image.size
        max_dimension, min_dimension = max(old_size), min(old_size)
        desired_size = (max_dimension, max_dimension)
        position = int(max_dimension/2) - int(min_dimension/2)

        if self.fill=="gaussian":
            # fill = np.random.normal(0, 3, (max_dimension, max_dimension, 3))
            # padded_image = Image.fromarray(fill.astype('uint8'), 'RGB')
            stat = ImageStat.Stat(image)
            R = np.random.normal(stat.mean[0], stat.stddev[0], (max_dimension, max_dimension, 1))
            G = np.random.normal(stat.mean[1], stat.stddev[1], (max_dimension, max_dimension, 1))
            B = np.random.normal(stat.mean[2], stat.stddev[2], (max_dimension, max_dimension, 1))
            fill = np.clip(np.dstack((R,G,B)), 0, 255)

            padded_image = Image.fromarray(fill.astype('uint8'), 'RGB')
            if self.smooth:
                padded_image = padded_image.filter(ImageFilter.GaussianBlur(radius = self.smooth))
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
        resize_to_min_size: int = 512,
        pad_to_square: bool = False,
        out_size: int = 512,
        center_crop: bool = False,
        horizontal_flip: float = None,
        color_jitter: Iterable = None,
        random_equalize: float = None,
        trivialwide: bool = False,
        to_tensor: bool = True,
        normalize: bool = True,
    ):
        self.output_dir = output_dir
        self.pad_to_square = pad_to_square
        self.resize_to_min_size = resize_to_min_size
        self.out_size = out_size
        self.center_crop = center_crop
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.random_equalize = random_equalize
        self.trivialwide = trivialwide
        
        self.interp_method = transforms.InterpolationMode.BILINEAR

        self.to_tensor = to_tensor
        self.normalize = normalize

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

        self.build()

    def build(self):
        augment_list = []
        
        if self.pad_to_square:
            augment_list.append(pad_to_square("gaussian"))            

        if self.resize_to_min_size is not None:
            augment_list.append(transforms.Resize(self.resize_to_min_size, interpolation=self.interp_method))

        if self.center_crop:
            augment_list.append(transforms.CenterCrop(self.out_size))
        else:
            augment_list.append(transforms.RandomCrop(self.out_size))

        if self.color_jitter:
            augment_list.append(transforms.ColorJitter(*self.color_jitter))

        if self.random_equalize:
            augment_list.append(transforms.RandomEqualize(self.random_equalize))

        if self.horizontal_flip:
            augment_list.append(transforms.RandomHorizontalFlip(self.horizontal_flip))
            
        if self.trivialwide:
            augment_list.append(transforms.TrivialAugmentWide())

        if len(augment_list)>0:
            self.augment = transforms.Compose(augment_list)
        else:
            self.augment = None
            
        # Convert to format usable by model. 
        # Keep separate in case dumping augmentations to disk
        transform_list = []
        if len(augment_list)==0:
            # Guard against images that have size != what the model is expecting
            transform_list.append(transforms.Resize(self.out_size, interpolation=self.interp_method))

        if self.to_tensor:
            transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

        if len(transform_list)>0:
            self.image_transforms = transforms.Compose(transform_list)
        else:
            self.image_transforms = None
        
    def __call__(self, image):
        if isinstance(image, str):
            image_path = Path(image)
            image = Image.open(image_path)

        elif isinstance(image, Path):
            image_path = image
            image = Image.open(image_path)

        if image.mode != "RGB":
            image = image.convert("RGB")
            
        if self.augment:
            image = self.augment(image)
            if self.output_dir is not None:
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = image_path.stem + f"-{hash_image}.jpg"
                image.save(os.path.join(self.output_dir, image_filename))
                
        if self.image_transforms:
            image = self.image_transforms(image)

        return image
