# Fine-tuning of Stable Diffusion models


Run [Dreambooth](https://arxiv.org/abs/2208.12242) or [Low-rank Adaptation (LoRA)](https://github.com/cloneofsimo/lora) from the same notebook:

<a target="_blank" href="https://colab.research.google.com/github/brian6091/Dreambooth/blob/main/FineTuning_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" height="28px" width="162px" alt="Open In Colab"/>
</a>


$~$


Tested with Tesla T4 and A100 GPUs on Google Colab (some settings will not work on T4 due to limited memory)

Tested with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [Stable Diffusion v2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base).

This notebook one borrows elements from [ShivamShrirao's](https://github.com/ShivamShrirao/diffusers) implementation, but is distinguished by some additional features:
* based on [Hugging Face](https://huggingface.co/) [Diffusers🧨](https://github.com/huggingface/diffusers) implementation so it's easy to stay up-to-date
* Low-rank Adaptation (LoRA) for fast text-to-image fine-tuning (using [cloneofsimo's implementation](https://github.com/cloneofsimo/lora))
* Data augmentation such as random cropping, flipping and resizing, which let you avoid having to manually prep and crop images in certain cases (e.g., training a style)
* exposes lesser-explored parameters for experimentation ([modify rank of LoRA approximation](https://github.com/cloneofsimo/lora/discussions/37), ADAM optimizer parameters, [cosine_with_restarts](https://huggingface.co/transformers/v2.9.1/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup) learning rate scheduler, etc), all of which are dumped to a json file so you can remember what you did
* possibility to drop some text-conditioning to improve classifier-free guidance sampling (e.g., how [SD V1-5 was fine-tuned](https://huggingface.co/runwayml/stable-diffusion-v1-5))
* image captioning using filenames or associated textfiles
* training loss and prior class loss are tracked separately (can be visualized using tensorboard)
* option to generate exponentially-weighted moving average (EMA) weights for the unet
* easily switch in different variational autoencoders (VAE) or text encoders
* inference with trained models is done using [Diffusers🧨](https://github.com/huggingface/diffusers) pipelines, does not rely on any web-apps

Image comparing Dreambooth and LoRA ([more information here](https://github.com/cloneofsimo/lora/discussions/37)):

<a><img src="https://drive.google.com/uc?id=1PQqL3omKCWStkrJgW3JecOrne3xqbScr"></a>
[full-size image here for the pixel-peepers](https://drive.google.com/file/d/16aQcDOg-DJ_1PB6ypzQAauaJEcbn0Vkx/view?usp=share_link "Comparison full-size")

[<a href="https://www.buymeacoffee.com/jvsurfsqv" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="45px" width="162px" alt="Buy Me A Coffee"></a>](https://www.buymeacoffee.com/jvsurfsqv)
