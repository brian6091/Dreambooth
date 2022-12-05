# Dreambooth-style fine tuning of Stable Diffusion models

[![Train In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brian6091/Dreambooth/blob/main/Dreambooth_colab.ipynb)

A notebook for training Stable Diffusion models 

There are lots of notebooks for Dreambooth-style training. This one is distinguished by a couple of features:
* based on [Hugging Face](https://huggingface.co/) [Diffusers🧨](https://github.com/huggingface/diffusers) implementation so it's easy to stay up-to-date with rapid access to new features
* exposes lesser-known parameters for experimentation 
* track/visualize training loss and prior class loss
* easily switch in different variational autoencoders (VAE) or text encoders
* option to generate exponentially-weighted moving average (EMA) weights
* inference with trained models is done using [Diffusers🧨](https://github.com/huggingface/diffusers) pipelines, does not rely on any web-apps


[<a href="https://www.buymeacoffee.com/anzorq" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" height="45px" width="162px" alt="Buy Me A Coffee"></a>](https://www.buymeacoffee.com/jvsurfsqv)
