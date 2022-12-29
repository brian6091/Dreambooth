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
#    Based on https://github.com/huggingface/diffusers/blob/v0.8.0/examples/dreambooth/train_dreambooth.py
#    SPDX short identifier: Apache-2.0
import hashlib
import math
import random
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from .textual_inversion_templates import object_templates, style_templates

class FineTuningDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        tokenizer,
        add_instance_token=False,
        instance_data_root=None,
        instance_token=None,
        instance_prompt=None,
        prompt_templates=None,
        class_data_root=None,
        class_prompt=None,
        use_image_captions=False,
        unconditional_prompt=" ",
        size=512,
        augment_output_dir=None,
        augment_min_resolution=None,
        augment_center_crop=False,
        augment_hflip=False,
        debug=False,
    ):
        self.tokenizer = tokenizer
        self.add_instance_token = add_instance_token
        self.instance_data_root = Path(instance_data_root)
        
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.instance_images_path = [path for path in self.instance_data_root.glob('*') if '.txt' not in path.suffix]
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.instance_token = instance_token
        self.instance_prompt = instance_prompt.replace("{}", instance_token)

        if prompt_templates==None:
            self.prompt_templates = None
        elif prompt_templates=="object":
            self.prompt_templates = object_templates
        elif prompt_templates=="style":
            self.prompt_templates = style_templates
        else:
            raise ValueError(
                f"{args.prompt_templates} is not a known set of prompt templates."
            )        
        
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)

            self.class_images_path = [path for path in self.class_data_root.glob('*') if '.txt' not in path.suffix]
            random.shuffle(self.class_images_path)
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.use_image_captions = use_image_captions
        self.unconditional_prompt = unconditional_prompt
        
        self.size = size
        self.augment_output_dir = augment_output_dir
        if augment_output_dir is not None:
            os.makedirs(augment_output_dir, exist_ok=True)
        self.augment_min_resolution = augment_min_resolution
        self.augment_center_crop = augment_center_crop
        self.augment_hflip = augment_hflip

        self.debug = debug
        
        # Data augmentation pipeline
        augment_list = []
        if augment_min_resolution is not None:
            augment_list.append(transforms.Resize(augment_min_resolution))
        if augment_center_crop:
            augment_list.append(transforms.CenterCrop(size))
        else:
            augment_list.append(transforms.RandomCrop(size))
        if augment_hflip:
            augment_list.append(transforms.RandomHorizontalFlip(0.5))

        # Convert to format usable by model. 
        # Keep separate in case dumping augmentations to disk
        transform_list = []
        transform_list.append(transforms.ToTensor())
        # check that this matches for textual inversion image = (image / 127.5 - 1.0).astype(np.float32)
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        
        if len(augment_list)>0:
            self.augment_transforms = transforms.Compose(augment_list)
        else:
            self.augment_transforms = None
            
        self.image_transforms = transforms.Compose(transform_list)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        image_path = self.instance_images_path[index % self.num_instance_images]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.augment_transforms is not None:
            image = self.augment_transforms(image)
            if self.augment_output_dir is not None:
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = image_path.stem + f"-{hash_image}.jpg"
                image.save(os.path.join(self.augment_output_dir, image_filename))
        example["instance_images"] = self.image_transforms(image)

        if self.prompt_templates is not None:
            self.instance_prompt = random.choice(self.prompt_templates).format(self.instance_token)
        elif self.use_image_captions:
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists():
                with open(caption_path) as f:
                    caption = f.read()
            else:
                caption = caption_path.stem
                
            caption = ''.join([i for i in caption if not i.isdigit()]) # not sure necessary
            caption = caption.replace("_"," ")
            self.instance_prompt = caption
        else:
            if self.instance_prompt is None:
                raise ValueError("An instance_prompt must be provided if prompt templates not provided and use_image_captions=False.")

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        if self.debug:
            print("\nInstance: " + str(image_path))
            print(self.instance_prompt)
            print(example["instance_prompt_ids"])

        if self.class_data_root:
            image_path = self.class_images_path[index % self.num_class_images]
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            if self.augment_transforms is not None:
                image = self.augment_transforms(image)
                if self.augment_output_dir is not None:
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = image_path.stem + f"-{hash_image}.jpg"
                    image.save(os.path.join(self.augment_output_dir, image_filename))
            example["class_images"] = self.image_transforms(image)
            
            if self.use_image_captions:
                caption_path = image_path.with_suffix(".txt")
                if caption_path.exists():
                    with open(caption_path) as f:
                        caption = f.read()
                else:
                    # Take filename as caption
                    caption = caption_path.stem
                    caption = caption.replace("_"," ")

                #caption = ''.join([i for i in caption if not i.isdigit()]) # not sure necessary
                caption = caption.replace("{}", self.instance_token)
                self.class_prompt = caption
            
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            
            if self.debug:
                print("\nClass: " + str(image_path))
                print(self.class_prompt)

        example["unconditional_prompt_ids"] = self.tokenizer(
                self.unconditional_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

    
def collate_fn(examples,
               with_prior_preservation=False,
               conditioning_dropout_prob=0.0,
               debug=False,
               
):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    # Apply text-conditioning dropout by inserting uninformative prompt
    if conditioning_dropout_prob > 0:
        for i, input_id in enumerate(input_ids):
            if random.uniform(0.0, 1.0) <= conditioning_dropout_prob:
                input_ids[i] = example["unconditional_prompt_ids"]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    input_ids = torch.cat(input_ids, dim=0)

    if debug:
        print("in collate_fn")
        print(input_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return 
