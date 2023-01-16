import os
import torch
from tqdm.auto import tqdm

import wandb

from accelerate.utils import (
    LoggerType,
    is_aim_available,
    is_comet_ml_available,
    is_mlflow_available,
    is_tensorboard_available,
    is_wandb_available,
)

from .utils import image_grid


def get_intermediate_samples(
    accelerator,
    pipeline,
    instance_token,
    sample_prompt,
    sample_negative_prompt,
    sample_guidance_scale,
    sample_infer_steps,
    sample_seed,
    save_n_sample,
    save_dir,
    sample_to_tracker,
    tracker,
    data_table,
    step,
):
    sample_prompt = sample_prompt.replace("{}", instance_token)
    sample_prompt = list(map(str.strip, sample_prompt.split('//')))

    g_cuda = torch.Generator(device=accelerator.device).manual_seed(sample_seed)
    pipeline.set_progress_bar_config(disable=True)
    sample_dir = os.path.join(save_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    with torch.autocast("cuda"), torch.inference_mode():
        all_images = []
        for sample_id in tqdm(range(save_n_sample), desc="Generating samples"):
            images = pipeline(
                sample_prompt,
                negative_prompt=[sample_negative_prompt]*len(sample_prompt),
                guidance_scale=sample_guidance_scale,
                num_inference_steps=sample_infer_steps,
                generator=g_cuda
            ).images
            all_images.extend(images)

            if sample_to_tracker:
                if tracker=="wandb" and is_wandb_available():
                    for prompt_id, im in enumerate(images):
                        data_table.add_data(step, prompt_id, sample_prompt[j], sample_guidance_scale,
                                           sample_seed, sample_id, wandb.Image(im))

        grid = image_grid(all_images, rows=save_n_sample, cols=len(sample_prompt))
        
        if sample_to_tracker:
            if tracker=="wandb" and is_wandb_available():
                accelerator.log({"sample_grid":[wandb.Image(grid, caption="test")]}, step=step)
        
        grid.save(os.path.join(sample_dir, f"{step}.jpg"), quality=90, optimize=True)

    return data_table
