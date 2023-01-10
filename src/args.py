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
#
import os
import configargparse
import yaml
import ast

def none_or_str(val):
    if not val or (val=='None'):
        return None
    return val

def none_or_int(val):
    if not val or (val=='None'):
        return None
    return int(val)

def none_or_float(val):
    if not val or (val=='None'):
        return None
    return float(val)

def none_or_set(val):
    if val!=None:
        return set(val)

# https://stackoverflow.com/a/42355279
class StoreDictKeyPair(configargparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(";"):
             k,v = kv.split("=")
             my_dict[k] = ast.literal_eval(v)
         setattr(namespace, self.dest, my_dict)

def parse_args(input_args=None):
    parser = configargparse.ArgParser(
        description='finetune.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    
    parser.add_argument(
        '-c',
        '--config',
        required=False,
        is_config_file=True,
        help='config file path',
        type=yaml.safe_load,
    )
    
    parser.add_argument(
        "--description",
        type=none_or_str,
        default=None,
        help="Description",
    )
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=none_or_str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=none_or_str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_tokenizer_name_or_path",
        type=none_or_str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--revision",
        type=none_or_str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIMScheduler",
        help="Scheduler",
    )
    parser.add_argument(
        "--scheduler_config",
        dest="scheduler_config",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1;KEY2=VAL2...",
        help="Scheduler parameters as semi-colon separated string",
        )
    
    parser.add_argument(
        "--train_unet_module_or_class",
        nargs='+',
        help="Modules or classes of the Unet to train.",
    )
    parser.add_argument(
        "--train_unet_submodule",
        nargs='+',
        default=None,
        help="Parameters of the Unet to train.",
    )
    parser.add_argument(
        "--train_text_module_or_class",
        nargs='+',
        help="Modules or classes of the text encoder to train.",
    )
    parser.add_argument(
        "--train_text_submodule",
        nargs='+',
        default=None,
        help="Parameters of the text encoder to train.",
    )
    
    parser.add_argument(
        "--lora_unet_layer",
        nargs='+',
        default=None,
        help="Layer to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_unet_train_off_target",
        nargs='+',
        default=None,
        help="Set defining classes to enable when LoRA cannot be injected while traversing Unet model.",
    )
    parser.add_argument(
        "--lora_unet_rank",
        type=none_or_int,
        default=4,
        help="Rank reduction for LoRA.",
    )
    parser.add_argument(
        "--lora_unet_scale",
        type=none_or_float,
        default=4.0,
        help="Scale for LoRA in Unet.",
    )
    parser.add_argument(
        "--lora_unet_nonlin",
        type=none_or_str,
        default=None,
        help="Nonlinearity for LoRA in Unet.",
    )
    
    parser.add_argument(
        "--lora_text_layer",
        nargs='+',
        default=None,
        help="Layer to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_text_train_off_target",
        nargs='+',
        default=None,
        help="Set defining classes to enable when LoRA cannot be injected while traversing text encoder model.",
    )
    parser.add_argument(
        "--lora_text_rank",
        type=none_or_int,
        default=4,
        help="Rank reduction for LoRA.",
    )
    parser.add_argument(
        "--lora_text_scale",
        type=none_or_float,
        default=4.0,
        help="Scale for LoRA in text encoder.",
    )
    parser.add_argument(
        "--lora_text_nonlin",
        type=none_or_str,
        default=None,
        help="Nonlinearity for LoRA in text encoder.",
    )
    
    parser.add_argument(
        "--add_instance_token",
        action="store_true",
        help="Whether to add instance token to tokenizer dictionary",
    )

    parser.add_argument(
        "--instance_token",
        type=str,
        default=None,
        required=True,
        help="The token identifier specifying the instance",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--prompt_templates",
        type=str,
        default=None,
        help="Which prompt templates to use, object, style, or None",
    )
    
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_token",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling class images.",
    )
    
    parser.add_argument(
        "--use_instance_image_captions",
        action="store_true",
        help="Use instance image captions from textfile, otherwise filename",
    )
    parser.add_argument(
        "--use_class_image_captions",
        action="store_true",
        help="Use class image captions from textfile, otherwise filename",
    )

    parser.add_argument(
        "--conditioning_dropout_prob",
        type=none_or_float,
        default=0.0,
        help="Probability that conditioning is dropped.",
    )
    parser.add_argument("--unconditional_prompt",
        type=str,
        default=" ",
        help="Prompt for conditioning dropout.",
    )
    parser.add_argument(
        "--clip_skip",
        type=none_or_int,
        default=None,
        help="Clip skip, to be implemented.",
    )
    
    parser.add_argument(
        "--augment_output_dir",
        type=none_or_str,
        default=None,
        help="The output directory where the image data augmentations will be saved.",
    )
    parser.add_argument(
        "--augment_min_resolution",
        type=none_or_int,
        nargs='?',
        default=None,
        help="Resize minimum image dimension before augmention pipeline.",
    )
    parser.add_argument(
        "--augment_center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--augment_hflip",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--augment_trivialwide",
        action="store_true",
        help="Whether to use TrivialAugment Wide",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--enable_full_determinism",
        action="store_true",
        help="Enable fullly deterministic runs by disabling some cuda features. May lower performance.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all images will be resized to this resolution.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int, default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Whether or not to enable xformers.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 for faster training on Ampere GPUs.",
    )

    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW8bit",
        help="Optimizer",
    )
    parser.add_argument(
        "--optimizer_params",
        dest="optimizer_params",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1;KEY2=VAL2...",
        help="Optimizer parameters as semi-colon separated string",
        )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    

    parser.add_argument(
        "--lr_scale",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_unet",
        type=none_or_float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use for Unet parameters.",
    )
    parser.add_argument(
        "--lr_text",
        type=none_or_float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use for text encoder.",
    )
    parser.add_argument(
        "--separate_token_embedding",
        action="store_true",
        help="If token embedding is trainable, separate into isolated parameter group?",
    )
    parser.add_argument(
        "--lr_token_embedding",
        type=none_or_float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use for token embedding if separate_token_embedding=True.",
    )
    
    parser.add_argument(
        "--lr_scheduler_params",
        dest="lr_scheduler_params",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1;KEY2=VAL2...",
        help="Learning rate scheduler parameters as semi-colon separated string",
        )
    
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use EMA model.",
    )
    parser.add_argument("--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma parameter for the EMA model.",
    )
    parser.add_argument("--ema_power",
        type=float,
        default=3 / 4,
        help="Exponential factor of EMA warmup.",
    )
    parser.add_argument("--ema_min_value",
        type=float,
        default=0.0,
        help="The minimum EMA decay rate.",
    )
    parser.add_argument("--ema_max_value",
        type=float,
        default=0.9999,
        help="The maximum EMA decay rate.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    
    parser.add_argument(
        "--save_n_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10_000,
        help="Save weights every N steps.",
    )
    parser.add_argument(
        "--save_min_steps",
        type=int, default=0,
        help="Start saving weights after N steps.",
    )
    parser.add_argument(
        "--save_parameter_summary",
        action="store_true",
        help="Save a summary of parameters that were set to be trained.",
    )
    parser.add_argument(
        "--save_model_layout",
        action="store_true",
        help="Save model hierarchy of Unet and text encoder (requires torchinfo library).",
    )
    
    parser.add_argument(
        "--sample_scheduler",
        type=str,
        default="DDIMScheduler",
        help="Sample scheduler",
    )
    parser.add_argument(
        "--sample_scheduler_config",
        dest="sample_scheduler_config",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1;KEY2=VAL2...",
        help="Sample scheduler parameters as semi-colon separated string",
        )
    parser.add_argument(
        "--sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--sample_seed",
        type=none_or_int,
        default=None,
        help="A seed for intermediate samples.",
    )
    parser.add_argument(
        "--sample_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--sample_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )

    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    
    parser.add_argument(
        "--log_gpu",
        action="store_true",
        help="Whether or not to log GPU memory usage.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Some exra verbosity."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
        
    args.train_unet_module_or_class = none_or_set(args.train_unet_module_or_class)
    args.train_unet_submodule = none_or_set(args.train_unet_submodule)
    args.train_text_module_or_class = none_or_set(args.train_text_module_or_class)
    args.train_text_submodule = none_or_set(args.train_text_submodule)
    args.lora_unet_layer = none_or_set(args.lora_unet_layer)
    args.lora_unet_train_off_target = none_or_set(args.lora_unet_train_off_target)
    args.lora_text_layer = none_or_set(args.lora_text_layer)
    args.lora_text_train_off_target = none_or_set(args.lora_text_train_off_target)

    return args

def format_args(args):
    d = args.__dict__.copy()
    
    for keys in d:
      if isinstance(d[keys], set):
          d[keys] = list(d[keys])
      elif isinstance(d[keys], dict):
          tmp = []
          d2 = d[keys]
          for k in d2:
              tmp.append(k+"="+str(d2[k]))
          d[keys] = ';'.join(tmp)

    del d['config']
    return d
