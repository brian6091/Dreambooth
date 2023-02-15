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
#    save_weights intially from https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
#    SPDX short identifier: Apache-2.0
#
import yaml
import hashlib
import itertools
import math
import os
from pathlib import Path
import inspect

import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    set_seed,
    LoggerType,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers import AutoencoderKL, DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, enable_full_determinism
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import Repository

from tqdm.auto import tqdm
from transformers import CLIPTextModel, AutoTokenizer

from lora_diffusion import tune_lora_scale

from src.datasets import FinetuneTrainDataset, PromptDataset, collate_fn
from src.augment import Augmentor
from src.args import parse_args, format_args
from src.model_utils import (
    SAFE_CONFIGS,
    get_noise_scheduler,
    add_instance_tokens,
    set_trainable_parameters,
    count_parameters,
    print_trainable_parameters,
    get_pipeline,
    get_module_by_name,
    save_trainable_parameters,
)
from src.optim import (
    calculate_loss,
    get_optimizer, 
    group_parameters,
    get_explore_exploit_schedule_with_warmup,
)
from src.tracking import get_intermediate_samples
from src.utils import image_grid, get_full_repo_name, get_gpu_memory_map


logger = get_logger(__name__)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    # TODO: CHECK FATAL ERRORS (e.g., no training, etc)
    # TODO: check instance and class (if given) path existence
            
    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
        
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.tracker,
        logging_dir=logging_dir,
    )
    
    # https://huggingface.co/docs/diffusers/optimization/fp16
    if args.enable_autotuner:
        torch.backends.cudnn.benchmark = True
    if args.allow_tf32:
        #torch.backends.cudnn.allow_tf32 = True (this is True by default)
        # TODO make parameter, as full f32 training is affected
        torch.backends.cuda.matmul.allow_tf32 = True
        
    if args.seed is not None:
        if args.enable_full_determinism:
            enable_full_determinism(args.seed)
        else:
            set_seed(args.seed)
            
    if args.debug:
        torch.set_printoptions(precision=10)

    if accelerator.is_main_process:
        # Handle the repository creation
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        
        # TODO clean up saving and sample generation to account for when there is no output_dir
        # e.g., everything is pushed to tracker or hub?
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            
        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.dump(format_args(args), f, indent=2, sort_keys=False)            
            
    # TODO move to function or at least use get_pipeline, move under main process above
    # Put after component loading, so that we can at least use the components to make pipeline
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        # Generate class images if necessary
        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae",
                    revision=None if args.pretrained_vae_name_or_path else args.revision,
                    torch_dtype=torch_dtype,
                ),
                torch_dtype=torch_dtype, # TODO allow selection?
                safety_checker=None,
                requires_safety_checker=None,
                revision=args.revision
            )
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to(accelerator.device)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Load diffusion components
    if args.pretrained_tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_tokenizer_name_or_path,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
        
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    
    if args.add_instance_token:
        instance_token_id, initializer_token_id = add_instance_tokens(
            tokenizer,
            text_encoder,
            instance_token=args.instance_token,
            initializer_token=args.initializer_token,
            debug=args.debug,
        )

    vae = AutoencoderKL.from_pretrained(        
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    
    unet.requires_grad_(False)
    set_trainable_parameters(unet,
                             target_module_or_class=args.train_unet_module_or_class,
                             target_submodule=args.train_unet_submodule,
                             lora_layer=args.lora_unet_layer,
                             lora_rank=args.lora_unet_rank,
                             lora_scale=args.lora_unet_scale,
                             lora_nonlin=args.lora_unet_nonlin,
                             lora_init=args.lora_unet_init,
                             lora_train_off_target=args.lora_unet_train_off_target)
        
    text_encoder.requires_grad_(False)
    set_trainable_parameters(text_encoder,
                             target_module_or_class=args.train_text_module_or_class,
                             target_submodule=args.train_text_submodule,
                             lora_layer=args.lora_text_layer,
                             lora_rank=args.lora_text_rank,
                             lora_scale=args.lora_text_scale,
                             lora_nonlin=args.lora_text_nonlin,
                             lora_init=args.lora_text_init,
                             lora_train_off_target=args.lora_text_train_off_target)

    if args.save_parameter_summary:
        f = open(os.path.join(args.output_dir, "unet_trainable_parameters.txt"), "w")
        count_parameters(unet, file=f)
        print_trainable_parameters(unet, file=f)
        f.close()
        f = open(os.path.join(args.output_dir, "text_encoder_trainable_parameters.txt"), "w")
        count_parameters(text_encoder, file=f)
        print_trainable_parameters(text_encoder, file=f)
        f.close()
    
    if args.save_model_layout:
        try:
            from torchinfo import summary
            
            with open(os.path.join(args.output_dir, "unet_layout.txt"), "w") as f:
                f.write(str(summary(unet, col_names=["num_params", "trainable"], verbose=2)))
                f.close()
            with open(os.path.join(args.output_dir, "text_encoder_layout.txt"), "w") as f:
                f.write(str(summary(text_encoder, col_names=["num_params", "trainable"], verbose=2)))
                f.close()
        except:
            print("pip install torchinfo if you want to save model layouts.")        

    
    lr_scaling = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    train_token_embedding, train_text_encoder, train_unet, params_to_optimize = group_parameters(
        unet,
        args.lr_unet,
        text_encoder,
        args.lr_text,
        lr_scaling=lr_scaling if args.lr_scale else 1.0,
        separate_token_embedding=args.separate_token_embedding,
        lr_token_embedding=args.lr_token_embedding,
        debug=args.debug,
    )
    
    if len(params_to_optimize)==0:
        raise ValueError("This configuration does not train anything.")
    
    if train_unet and args.enable_xformers and is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            # TODO vae.enable_xformers_memory_efficient_attention() is inherited?
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
            
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    if (train_unet and (train_text_encoder or train_token_embedding)) and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        logger.warning(
            "Gradient accumulation is not supported when training both unet and the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.gradient_checkpointing:
        if train_unet:
            unet.enable_gradient_checkpointing()
        if train_token_embedding or train_text_encoder:
            logger.warning("Gradient checkpointing for the text_encoder not implemented")
            # https://github.com/brian6091/Dreambooth/issues/23
            #text_encoder.gradient_checkpointing_enable()

    optimizer_class = get_optimizer(args.optimizer)
        
    optimizer_params = args.optimizer_params
    optimizer_params["params"] = params_to_optimize
    optimizer = optimizer_class(**optimizer_params)
    if True:#args.debug: # TODO remove
        print(optimizer)
    
    # Set up scheduler for training
    if args.scheduler and args.scheduler_config:
        noise_scheduler = get_noise_scheduler(args.scheduler, config=args.scheduler_config)        
    elif args.scheduler:
        noise_scheduler = get_noise_scheduler(args.scheduler, model_name_or_path=args.pretrained_model_name_or_path)
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.debug:
        print(noise_scheduler.__class__.__name__)
        print(noise_scheduler.config)
        
    if noise_scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    train_dataset = FinetuneTrainDataset(
        tokenizer=tokenizer,
        add_instance_token=args.add_instance_token,
        instance_data_root=args.instance_data_dir,
        instance_token=args.instance_token,
        instance_prompt=args.instance_prompt,
        prompt_templates=args.prompt_templates,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        use_instance_image_captions=args.use_instance_image_captions,
        use_class_image_captions=args.use_class_image_captions,
        unconditional_prompt=args.unconditional_prompt,
        size=args.resolution,
        augmentor=Augmentor(**args.augmentation) if args.augmentation else Augmentor(),
        debug=args.debug,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples,
            args.with_prior_preservation,
            args.conditioning_dropout_prob,
            args.debug,
        ),
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.lr_scheduler in ("explore_exploit", "multi_explore_exploit"):
        # TODO multiply tuples by gradient_accumulation_steps
        # TODO sensible defaults if no dict passed in?
        lr_scheduler =  get_explore_exploit_schedule_with_warmup(
            optimizer,
            start_step=args.lr_scheduler_params["start_step"],
            num_warmup_steps=args.lr_scheduler_params["num_warmup_steps"],
            num_explore_steps=args.lr_scheduler_params["num_explore_steps"],
            num_total_steps=args.lr_scheduler_params["num_total_steps"],
            plateau=args.lr_scheduler_params["plateau"] if "plateau" in args.lr_scheduler_params.keys() else None,
        )
    else:
        lr_scheduler_params = args.lr_scheduler_params
        lr_scheduler_params["name"] = args.lr_scheduler
        lr_scheduler_params["optimizer"] = optimizer
        if "num_warmup_steps" in lr_scheduler_params:
            lr_scheduler_params["num_warmup_steps"] = lr_scheduler_params["num_warmup_steps"] * args.gradient_accumulation_steps
        else:
            lr_scheduler_params["num_warmup_steps"] = 0
        if "num_training_steps" in lr_scheduler_params:
            lr_scheduler_params["num_training_steps"] = lr_scheduler_params["num_training_steps"] * args.gradient_accumulation_steps
        else:
            lr_scheduler_params["num_training_steps"] = args.max_train_steps * args.gradient_accumulation_steps
            
        lr_scheduler = get_scheduler(**lr_scheduler_params)

        
    if train_unet and (train_text_encoder or train_token_embedding):
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    elif train_unet:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    else:
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move untrained components to GPU
    # For mixed precision training we cast the weights to half-precision as it they are 
    # only used for inference, and keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    if not train_unet:
        unet.to(accelerator.device, dtype=weight_dtype)
        unet.eval()
    if not (train_text_encoder or train_token_embedding):
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.eval()

    # https://github.com/huggingface/diffusers/issues/1566
    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
        inspect.signature(accelerator.unwrap_model).parameters.keys()
    )
    unwrap_args = (
        {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
    )

    # Create EMA for the unet.
    if args.use_ema and train_unet:
        ema_unet = EMAModel(
            accelerator.unwrap_model(unet, **unwrap_args),
            inv_gamma=args.ema_inv_gamma, # TODO dictionary inputs
            power=args.ema_power, 
            min_value=args.ema_min_value,
            max_value=args.ema_max_value
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # Initialize the trackers (automatically on the main process)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            init_kwargs=args.tracker_init_kwargs,
        )
        
        if is_wandb_available():
            import wandb
            
            artifact = wandb.Artifact(name='config', type='configuration')
            artifact.add_file(local_path=os.path.join(args.output_dir, "args.yaml"), name='config.yaml')
            
            if args.save_parameter_summary:
                artifact.add_file(
                    local_path=os.path.join(args.output_dir, "text_encoder_trainable_parameters.txt"), 
                    name='text_encoder_trainable_parameters.txt')
                artifact.add_file(
                    local_path=os.path.join(args.output_dir, "unet_trainable_parameters.txt"), 
                    name='unet_trainable_parameters.txt')
                
            if args.save_model_layout:
                artifact.add_file(
                    local_path=os.path.join(args.output_dir, "text_encoder_layout.txt"), 
                    name='text_encoder_layout.txt')
                artifact.add_file(
                    local_path=os.path.join(args.output_dir, "unet_layout.txt"),
                    name='unet_layout.txt')
                
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.log_artifact(artifact)
            
            if args.tracker_watch:
                watch_modules = []
                for n in args.tracker_watch:
                    watch_modules.append(get_module_by_name(unet, n))

                wandb.watch(watch_modules, log='all', log_freq=10)
                # TODO paramter list of components to watch, otherwise all
                # need also log_freq and type as  parameters
                #wandb.watch()
                #wandb.watch((text_encoder, unet), log="all", log_freq=10)               
                
            if args.save_n_sample > 0:
                data_table = wandb.Table(columns=["step", "prompt_id", "prompt", "cfg", "seed", "sample", "image"])
        
    if args.evaluate:
        pipeline_evaluator = PipelineCLIPEvaluator(
            device=accelerator.device,
            target_images=args.instance_data_dir, # TODO, eventually could be a validation dataset?
            target_prompts=args.sample_prompt,
            instance_token=args.instance_token,
            class_token=args.class_token,
            clip_model='openai/clip-vit-large-patch14', # TODO make argument
        )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    # TODO: make this non-nested?
    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            save_dir = os.path.join(args.output_dir, f"{step}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        
            if args.lora_text_layer!=None or args.lora_unet_layer!=None:
                # TODO, this should activate when !ALL is trained, or should be a config flag save_full_model or save_diffusers_format
                # TODO optionally dump out keys to text file
                saved_keys = save_trainable_parameters(
                    tokenizer=tokenizer,
                    text_encoder=accelerator.unwrap_model(text_encoder, **unwrap_args),
                    unet=accelerator.unwrap_model(
                            ema_unet.averaged_model if args.use_ema else unet,
                            **unwrap_args,
                        ),
                    instance_token=args.instance_token if args.add_instance_token else None,
                    save_path=os.path.join(save_dir, f"{step}_trained_parameters.safetensors"),
                )
                
            print(f"[*] Weights saved at {save_dir}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    last_save_at_step = 0
    
    # TODO add sampling from base model step=0,
    
    if args.add_instance_token:
        orig_embeds_params = accelerator.unwrap_model(text_encoder, **unwrap_args).get_input_embeddings().weight.data.clone()

    for epoch in range(args.num_train_epochs):
        if train_unet:
            unet.train()
        if train_text_encoder or train_token_embedding:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if args.debug:
                print("\n", "\tEpoch: ", epoch, "step: ", step, "index: ", batch["indices"], "\n", '\n'.join(map(str, batch["image_paths"])),  "\n")
                
            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else: # "v_prediction"
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            if args.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                
                # Compute instance loss
                pred_loss = calculate_loss(model_pred.float(), target.float(), loss_function=args.loss, loss_adjust=args.loss_adjust)

                # Compute prior loss
                prior_loss = calculate_loss(model_pred_prior.float(), target_prior.float(), loss_function=args.loss, loss_adjust=args.loss_adjust)

                # Add the prior loss to the instance loss.
                loss = pred_loss + args.prior_loss_weight * prior_loss
            else:
                loss = calculate_loss(model_pred.float(), target.float(), loss_function=args.loss, loss_adjust=args.loss_adjust)

            loss = loss / args.gradient_accumulation_steps
            
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0:
                if accelerator.sync_gradients:
                    params_to_clip = itertools.chain(*[g["params"] for g in params_to_optimize])
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_unet.step(unet)
                optimizer.zero_grad()

                if args.add_instance_token and train_token_embedding:
                    # Re-insert original embedding weights for everything except the newly added token(s)
                    index_no_updates = torch.arange(len(tokenizer)) != instance_token_id
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder, **unwrap_args).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                eval_metrics = None
                if (global_step >= args.save_min_steps and not global_step % args.save_interval) or (global_step in args.save_at_steps):
                    # if accelerator.is_main_process:
                    # if args.lora_text_layer or args.lora_unet_layer:
                    # distinguish lora_text_layer/no_unet, text/unet, no_text/unet, ...
                    save_weights(global_step)
                    
                    save_dir = os.path.join(args.output_dir, f"{global_step}")
                    os.makedirs(save_dir, exist_ok=True)

                    last_save_at_step = global_step

                    # Set up scheduler for inference
                    sample_scheduler = None
                    if args.sample_scheduler and args.sample_scheduler_config:
                        sample_scheduler = get_noise_scheduler(args.sample_scheduler, config=args.sample_scheduler_config)        
                    elif args.sample_scheduler:
                        sample_scheduler = get_noise_scheduler(args.sample_scheduler, model_name_or_path=args.pretrained_model_name_or_path)

                    pipeline = get_pipeline(
                        args.pretrained_model_name_or_path,
                        vae=AutoencoderKL.from_pretrained(
                            args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                            subfolder=None if args.pretrained_vae_name_or_path else "vae",
                            revision=None if args.pretrained_vae_name_or_path else args.revision,
                        ),
                        tokenizer=tokenizer,
                        text_encoder=accelerator.unwrap_model(text_encoder, **unwrap_args) \
                            if (train_text_encoder or train_token_embedding) else text_encoder,
                        unet=accelerator.unwrap_model(unet, **unwrap_args) if train_unet else unet,
                        scheduler=sample_scheduler or noise_scheduler,
                        debug=True,
                        # TODO pass in torch_dtype from args.sample_precision?
                    )
                    
                    if args.evaluate or (args.save_n_sample > 0):
                        if args.lora_text_layer:
                            tune_lora_scale(pipeline.text_encoder, args.lora_text_scale)                        
                        if args.lora_unet_layer:
                            tune_lora_scale(pipeline.unet, args.lora_unet_scale)
                        
                        pipeline = pipeline.to(accelerator.device) # Nessecary? everything is on device already

                        if args.enable_xformers and is_xformers_available():
                            pipeline.enable_xformers_memory_efficient_attention()

                        grid, data_table, _, image_gen_instance = generate_samples(
                            pipeline=pipeline,
                            device=accelerator.device,
                            token=args.instance_token,
                            prompt=args.sample_prompt,
                            negative_prompt=args.sample_negative_prompt if args.sample_negative_prompt else '',
                            guidance_scale=args.sample_guidance_scale,
                            infer_steps=args.sample_infer_steps,
                            seed=args.sample_seed if args.sample_seed!=None else args.seed,
                            size=args.resolution,
                            n_samples=args.save_n_sample,
                            tracker=args.tracker,
                            data_table=data_table,
                            step=global_step,
                            make_grid=True if (args.save_n_sample > 0) or args.sample_to_tracker else False,
                        )

                        if args.evaluate:
                            eval_metrics = pipeline_evaluator.evaluate(
                                    pipeline=pipeline,
                                    image_gen_instance=image_gen_instance if args.save_n_sample > 0 else None,
                                    target_prompts=args.sample_prompt,
                                    n_samples=args.save_n_sample,
                                    negative_prompt=args.sample_negative_prompt if args.sample_negative_prompt else '',
                                    guidance_scale=args.sample_guidance_scale5,
                                    infer_steps=args.sample_infer_steps,
                                    seed=args.sample_seed if args.sample_seed!=None else args.seed,
                                    size=args.resolution,
                                    estimate_image_only=True,
                                    estimate_prompt_only=True,
                                    use_target_instance_cache=True,
                                )                            
                        
                        del pipeline
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        # TODO save_local parameter
                        if args.save_n_sample > 0:
                            sample_dir = os.path.join(save_dir, "samples")
                            os.makedirs(sample_dir, exist_ok=True)
                            grid.save(os.path.join(sample_dir, f"{global_step}.jpg"), quality=90, optimize=True)

                        if args.sample_to_tracker and args.tracker=="wandb" and is_wandb_available():
                            accelerator.log({"sample_grid":[wandb.Image(grid, caption=f"grid-step-{global_step}")]}, step=global_step)
                            
                        # args.checkpoints_to_tracker
                            
            # TODO function get_step_logs(pred_loss, prior_loss, loss, args)
            if args.with_prior_preservation:
                logs = {"Loss/pred": pred_loss.detach().item(),
                        "Loss/prior": prior_loss.detach().item(),
                        "Loss/total": loss.detach().item(),
                       }
            else:
                logs = {"Loss/pred": loss.detach().item()}

            if (train_token_embedding and args.separate_token_embedding) and train_text_encoder and train_unet:
                # TODO fix this to assign proper names https://discuss.pytorch.org/t/solved-use-two-scheduler-lambdalr-cosineannealinglr-but-seems-wierd/75184
                logs["lr/token"] = lr_scheduler.get_last_lr()[0]
                logs["lr/text"] = lr_scheduler.get_last_lr()[1]
                logs["lr/unet"] = lr_scheduler.get_last_lr()[2]
            elif train_text_encoder and train_unet:
                logs["lr/text"] = lr_scheduler.get_last_lr()[0]
                logs["lr/unet"] = lr_scheduler.get_last_lr()[1]
            elif train_token_embedding and train_unet:
                logs["lr/token"] = lr_scheduler.get_last_lr()[0]
                logs["lr/unet"] = lr_scheduler.get_last_lr()[1]
            elif train_token_embedding and train_text_encoder:
                logs["lr/token"] = lr_scheduler.get_last_lr()[0]
                logs["lr/text"] = lr_scheduler.get_last_lr()[1]
            elif train_token_embedding:
                logs["lr/token"] = lr_scheduler.get_last_lr()[0]
            elif train_text_encoder:
                logs["lr/text"] = lr_scheduler.get_last_lr()[0]
            elif train_unet:
                logs["lr/unet"] = lr_scheduler.get_last_lr()[0]
            else:
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                
            if args.log_gpu:
                logs["GPU"] = get_gpu_memory_map()[0]
                                
            if args.use_ema:
                logs["ema_decay"] = ema_unet.decay
                    
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            metrics = {}
            if eval_metrics:
                metrics[''] = # TODO
                accelerator.log(metrics, step=global_step)
                            
            if global_step >= args.max_train_steps:
                break
            
        accelerator.wait_for_everyone()
        #optimizer.epoch_step()

                            
    if accelerator.is_main_process:
        if args.sample_to_tracker:
            if args.tracker=="wandb" and is_wandb_available():
                accelerator.log({"samples": data_table}, step=step)
                
        if global_step > last_save_at_step:
            save_weights(global_step)
    
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)
            
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
