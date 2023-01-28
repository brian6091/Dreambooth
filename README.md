# One script to rule them all

Fine-tune Stable diffusion models using [Dreambooth](https://arxiv.org/abs/2208.12242), [Textual inversion](https://arxiv.org/abs/2208.01618), [Custom diffusion](https://arxiv.org/abs/2212.04488), [Low-rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685), and mixtures and variants thereof all in one place.

A quite skeletal notebook to get started:

<a target="_blank" href="https://colab.research.google.com/github/brian6091/Dreambooth/blob/main/FineTuning_config_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" height="28px" width="162px" alt="Open In Colab"/>
</a>

$~$

Tested with Tesla T4 and A100 GPUs on Google Colab (some configurations will not work on T4 due to limited memory)

Tested with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Stable Diffusion v2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base), and [Stable Diffusion v2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1).

Some unique features:
* **Advanced configuration:** Mix-and-match different fine-tuning methods ([LoRA](https://github.com/cloneofsimo/lora) X Dreambooth, Dreambooth X Textual inversion, etc)
* **Train only what you want:** 
* **Different loss criteria**
* **Continual learning**
* **Optimizer zoo:** In addition to the standard AdamW (and it's 8-bit variant), you can try out Dadaptation, 
* **Data augmentation:** such as random cropping, flipping and resizing, which can minimize manually prepping and cropping images in certain cases (e.g., training a style)
* Drop some text-conditioning to improve classifier-free guidance sampling (e.g., how [SD V1-5 was fine-tuned](https://huggingface.co/runwayml/stable-diffusion-v1-5))
* Image captioning using filenames or associated textfiles
* Multiple tokens for jointly training multiple concepts
* Inference with trained models uses [Diffusers🧨](https://github.com/huggingface/diffusers) pipelines, does not rely on any web-apps

$~$

Image comparing Dreambooth and LoRA ([more information here](https://github.com/cloneofsimo/lora/discussions/37)):

<a><img src="https://drive.google.com/uc?id=1PQqL3omKCWStkrJgW3JecOrne3xqbScr"></a>
[full-size image here for the pixel-peepers](https://drive.google.com/file/d/16aQcDOg-DJ_1PB6ypzQAauaJEcbn0Vkx/view?usp=share_link "Comparison full-size")

# Credits
This notebook was initially based on the Diffusers🧨 [example](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py), with elements from [ShivamShrirao's fork](https://github.com/ShivamShrirao/diffusers).

# Copyright

[<a href="https://www.buymeacoffee.com/jvsurfsqv" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="45px" width="162px" alt="Buy Me A Coffee"></a>](https://www.buymeacoffee.com/jvsurfsqv)

