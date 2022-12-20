# Fine-tuning of Stable Diffusion models


Run [Dreambooth](https://arxiv.org/abs/2208.12242), [Textual inversion](https://arxiv.org/abs/2208.01618), and/or [Low-rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685)-based fine-tuning from the same notebook:

<a target="_blank" href="https://colab.research.google.com/github/brian6091/Dreambooth/blob/main/FineTuning_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" height="28px" width="162px" alt="Open In Colab"/>
</a>

$~$

Tested with Tesla T4 and A100 GPUs on Google Colab (some settings will not work on T4 due to limited memory)

Tested with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [Stable Diffusion v2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base).

This notebook was initially inspired by [ShivamShrirao's](https://github.com/ShivamShrirao/diffusers) implementation, but has since evolved to include some unique features:
* Based on main [Hugging Face](https://huggingface.co/) [Diffusers🧨](https://github.com/huggingface/diffusers) so it's easy to stay up-to-date
* Mix-and-match different fine-tuning methods (LoRA X Dreambooth, Dreambooth X Textual inversion, etc)
* Low-rank Adaptation (LoRA) for faster and more efficient fine-tuning (using [cloneofsimo's implementation](https://github.com/cloneofsimo/lora))
* Data augmentation such as random cropping, flipping and resizing, which can minimize manually prepping and cropping images in certain cases (e.g., training a style)
* More parameters for experimentation ([modify LoRA rank approximation](https://github.com/cloneofsimo/lora/discussions/37), ADAM optimizer parameters, [cosine_with_restarts](https://huggingface.co/transformers/v2.9.1/main_classes/optimizer_schedules.html#transformers.get_cosine_with_hard_restarts_schedule_with_warmup) learning rate scheduler, etc), all of which are dumped to a json file so you can remember what you did
* Drop some text-conditioning to improve classifier-free guidance sampling (e.g., how [SD V1-5 was fine-tuned](https://huggingface.co/runwayml/stable-diffusion-v1-5))
* Image captioning using filenames or associated textfiles
* Training loss and prior class loss are tracked separately (can be visualized using tensorboard)
* Option to generate exponentially-weighted moving average (EMA) weights for the unet
* Inference with trained models uses [Diffusers🧨](https://github.com/huggingface/diffusers) pipelines, does not rely on any web-apps

$~$

Image comparing Dreambooth and LoRA ([more information here](https://github.com/cloneofsimo/lora/discussions/37)):

<a><img src="https://drive.google.com/uc?id=1PQqL3omKCWStkrJgW3JecOrne3xqbScr"></a>
[full-size image here for the pixel-peepers](https://drive.google.com/file/d/16aQcDOg-DJ_1PB6ypzQAauaJEcbn0Vkx/view?usp=share_link "Comparison full-size")

[<a href="https://www.buymeacoffee.com/jvsurfsqv" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="45px" width="162px" alt="Buy Me A Coffee"></a>](https://www.buymeacoffee.com/jvsurfsqv)
