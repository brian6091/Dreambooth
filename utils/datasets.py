import hashlib
import math
import random
import os
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FineTuningDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    
    Textual inversion:
    add_instance_token=True
    prompt_templates!="None"
    train_text_embedding=True (in main script)
    train_text_encoder=False
    no prior preservation
    
    Dreambooth:
    add_instance_token=False
    prompt_templates=ignored?
    train_text_embedding=True (in main script)
    train_text_encoder=True
    
    train_text_embedding and train_text_encoder should be exclusive?
    otherwise
    train_text_embedding_only and train_text_encoder
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
        self.instance_prompt = instance_prompt
        self.prompt_templates = prompt_templates
        
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
            if self.debug:
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = image_path.stem + f"-{hash_image}.jpg"
                image.save(os.path.join("/content/augment", image_filename))
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
                raise ValueError("An instance_prompt must be provided if use_textual_inversion_templates=False, and use_image_captions=False.")

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        if self.debug:
            print("\nInstance: " + str(image_path))
            print(self.instance_prompt)

        if self.class_data_root:
            image_path = self.class_images_path[index % self.num_class_images]
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            if self.augment_transforms is not None:
                image = self.augment_transforms(image)
                if self.debug:
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = image_path.stem + f"-{hash_image}.jpg"
                    image.save(os.path.join("/content/augment", image_filename))
            example["class_images"] = self.image_transforms(image)
            
            if self.use_image_captions:
                caption_path = image_path.with_suffix(".txt")
                if caption_path.exists():
                    with open(caption_path) as f:
                        caption = f.read()
                else:
                    caption = caption_path.stem

                caption = ''.join([i for i in caption if not i.isdigit()]) # not sure necessary
                caption = caption.replace("_"," ")
                self.class_prompt = caption
            
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            
            if self.debug:
                print("\nClass: " + str(image_path))
                print(self.class_prompt)

        example["unconditional_prompt_ids"] = self.tokenizer(
                self.unconditional_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

      
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
