import configargparse
import yaml

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
        "--train_unet",
        action="store_true",
        help="Whether to train the unet",
    )
    parser.add_argument(
        "--train_unet_attn_only",
        type=none_or_str,
        nargs='?',
        default=None,
        help="Only train attention layers of unet.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train all modules of the text encoder",
    )
    parser.add_argument(
        "--train_text_embedding_only",
        action="store_true",
        help="Whether to train only the text embedding module of text encoder",
    )
    
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Whether or not to use lora."
    )
    parser.add_argument(
        "--lora_unet_modules",
        nargs='+',
        help="Modules of the Unet to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_text_modules",
        nargs='+',
        help="Modules of the text encoder to apply LoRA to.",
    )
    parser.add_argument(
        "--lora_unet_rank",
        type=none_or_int,
        default=4,
        help="Rank reduction for LoRA.",
    )
    parser.add_argument(
        "--lora_text_rank",
        type=none_or_int,
        default=4,
        help="Rank reduction for LoRA.",
    )
    parser.add_argument(
        "--lora_unet_alpha",
        type=none_or_float,
        default=4.0,
        help="Alpha for LoRA in Unet.",
    )
    parser.add_argument(
        "--lora_text_alpha",
        type=none_or_float,
        default=4.0,
        help="Alpha for LoRA in text encoder.",
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
        "--use_image_captions",
        action="store_true",
        help="Get captions from textfile, otherwise filename",
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
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--lr_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_cosine_num_cycles",
        type=float, default=1.0,
        help="Number of cycles when using cosine_with_restarts lr scheduler.",
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
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_seed",
        type=none_or_int,
        default=None,
        help="A seed for intermediate samples.",
    )
    parser.add_argument(
        "--save_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
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
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
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

    return args
