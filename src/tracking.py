#    Copyright 2023 B. Lau, brian6091@gmail.com
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
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, Iterable

import torch
from tqdm.auto import tqdm

from accelerate.utils import (
    LoggerType,
    is_tensorboard_available,
    is_wandb_available,
)

from .utils import image_grid


def generate_samples(
    pipeline,
    device,
    token: str,
    prompt: Union[Iterable[str], str],
    negative_prompt: Union[Iterable[str], str],
    guidance_scale: float,
    infer_steps: int,
    seed: int,
    size: int = 512,
    n_samples: int = 1,
    tracker=None,
    data_table=None,
    step=None,
    make_grid=False,
):
    if isinstance(prompt, str):
        prompt = prompt.replace("{}", token).strip()
        n_prompts = 1
    else:
        prompt = [p.replace("{}", token).strip() for p in prompt]
        n_prompts = len(prompt)
    
    generator = torch.Generator(device=device)
    state = generator.get_state()
    pipeline.set_progress_bar_config(disable=True)

    with torch.inference_mode():
        all_images = []
        for sample_id in tqdm(range(n_samples), desc="Generating samples"):
            # Increment seed for each sample
            sample_seed = seed + sample_id
            generator = generator.manual_seed(sample_seed)
            
            # Generate latent, which will be the same for all prompts
            latent = torch.randn(
              (1, pipeline.unet.in_channels, size // 8, size // 8), # 64 for sd1, 96 for sd2-1
              generator = generator,
              device = device,
              dtype = pipeline.unet.dtype,
            )
            
            images = pipeline(
                prompt,
                negative_prompt=[negative_prompt]*len(prompt),
                guidance_scale=guidance_scale,
                num_inference_steps=infer_steps,
                latents=latent.repeat(n_prompts, 1, 1, 1),
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
                        sample_seed,
                        sample_id,
                        Image(im, caption=f"step:{step}, prompt_id:{prompt_id}, sample_id:{sample_id}")
                       )

        grid = None
        if make_grid:
            grid = image_grid(all_images, rows=n_samples, cols=len(prompt))
        
        # Reset the generator state
        generator.set_state(state)
        
    return grid, data_table, prompt, all_images
