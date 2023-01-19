import os
import torch
from tqdm.auto import tqdm

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
    pipeline,
    device,
    instance_token,
    prompt,
    negative_prompt,
    guidance_scale,
    infer_steps,
    seed,
    save_n_sample,
    save_dir,
    tracker,
    data_table,
    step,
):
    prompt = prompt.replace("{}", instance_token)
    prompt = list(map(str.strip, prompt.split('//')))

    g_cuda = torch.Generator(device=device).manual_seed(sample_seed)
    pipeline.set_progress_bar_config(disable=True)

    with torch.inference_mode():
        all_images = []
        for sample_id in tqdm(range(save_n_sample), desc="Generating samples"):
            images = pipeline(
                prompt,
                negative_prompt=[negative_prompt]*len(prompt),
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                generator=g_cuda
            ).images
            all_images.extend(images)

            if data_table and tracker=="wandb" and is_wandb_available():
                from wandb import Image
                for prompt_id, im in enumerate(images):
                    data_table.add_data(
                        step,
                        prompt_id,
                        prompt[prompt_id],
                        guidance_scale,
                        seed,
                        sample_id,
                        Image(im, caption=f"step:{step}, prompt_id:{prompt_id}, sample_id:{sample_id}")
                       )

        grid = image_grid(all_images, rows=save_n_sample, cols=len(prompt))
        
    return grid, data_table
