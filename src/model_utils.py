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
import sys
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch.nn as nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, AutoencoderKL
from lora_diffusion import LoraInjectedLinear

from safetensors.torch import safe_open
from safetensors.torch import save_file as safe_save
    

def get_tensor_info(tensor):
    info = []
    for name in ['is_leaf', 'requires_grad', 'retains_grad', 'grad_fn', 'grad']:
        info.append(f'{name}({getattr(tensor, name, None)})')
    return ' '.join(info)


def print_trainable_parameters(model: nn.Module, file=sys.stdout):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape, file=file)
            print(get_tensor_info(p), file=file)


def find_modules_by_name_or_class(
    model: nn.Module,
    target: Set[str],
):
    for fullname, module in model.named_modules():
        *path, name = fullname.split(".")
        if (module.__class__.__name__ in target) or (name in target):
            yield module.__class__.__name__, fullname, name, module
            

# From lora_diffusion, TODO: import
def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).
    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


# adapted from: https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# SPDX short identifier: Apache-2.0
def _inject_trainable_lora(
    model: nn.Module,
    target_name: str,
    r: int = 4,
    scale: float = 1.0,
    nonlin: nn.Module = None,
    train_off_target: Set[str] = None,
):
    """
    inject lora for nn.Linear into model.
    """

    if target_name==None:
        for p, n, m in _find_children(model):
            if not isinstance(p, LoraInjectedLinear):
                _inject_trainable_lora(
                    p,
                    target_name=n,
                    r=r,
                    scale=scale,
                    nonlin=nonlin,
                    train_off_target=train_off_target,
                )
    else:
        try:
            _child_module = model._modules[target_name]
        except:
            print(f"{target_name} not in module")
            return

        if _child_module.__class__.__name__ == "Linear":
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r=r,
                scale=scale,
                nonlin=nonlin,
                init=None,
            )
            
#             print(target_name)
#             print(_tmp.alpha)
#             print(_tmp.r)
#             print(_tmp.scale)
#             print(_tmp.nonlin)
#             print(_tmp)
            
            # Assign pretrained parameters
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias

            # Switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            model._modules[target_name] = _tmp

            model._modules[target_name].lora_up.weight.requires_grad = True
            model._modules[target_name].lora_down.weight.requires_grad = True            
        else:
            print(f"Cannot inject LoRA into {_child_module.__class__.__name__}")
            if train_off_target!=None:
                if _child_module.__class__.__name__ in train_off_target:
                    print(f"But {_child_module.__class__.__name__} was enabled by request")
                    model._modules[target_name].requires_grad_(True)
                else:
                    print(f"But {_child_module.__class__.__name__} was *not* enabled by request")


def get_nonlin(nonlin: str):
    if nonlin=="ReLU":
        return nn.ReLU(inplace=True)
    elif nonlin=="GELU":
        return nn.GELU()
    elif nonlin=="SiLU":
        return nn.SiLU(inplace=True)
    elif nonlin=="Mish":
        return nn.Mish(inplace=True)
    else:
        return None
    

def set_trainable_parameters(
    model: nn.Module,
    target_module_or_class: Set[str],
    target_submodule: Set[str],
    lora_rank: int = 4,
    lora_scale: float = 1.0,
    lora_nonlin: str = None,
    lora_layer: Set[str] = None,
    lora_train_off_target: Set[str] = None,
):

    if target_module_or_class is not None:
        if "ALL" in target_module_or_class:
            # Shorcut to training everything
            model.requires_grad_(True)
        else:
            for _c, _f, _n, _m in find_modules_by_name_or_class(
                model, target=target_module_or_class
            ):
                if target_submodule is None:
                    if lora_layer is None:
                        # Train everything in module
                        _m.requires_grad_(True)
                    else:
                        # Inject LoRA into all valid children
                        _inject_trainable_lora(
                            _m,
                            target_name=None,
                            r=lora_rank,
                            scale=lora_scale,
                            nonlin=get_nonlin(lora_nonlin),
                            train_off_target=lora_train_off_target,
                            )       
                else:
                    for __c, __f, __n, __m in find_modules_by_name_or_class(
                        _m, target=target_submodule
                    ):
                        if lora_layer is None:
                            # Train everything in submodule
                            __m.requires_grad_(True)
                        else:
                            # Inject LoRA into all valid children
                            _inject_trainable_lora(
                                _m,
                                target_name=__n,
                                r=lora_rank,
                                scale=lora_scale,
                                nonlin=get_nonlin(lora_nonlin),
                                train_off_target=lora_train_off_target,
                                )


def get_trainable_param_dict(
    model: nn.Module,
    validate=True,
):
    tensors_dict = {}
    metadata = {}

    for nc, c in model.named_children():
        for nm, m in c.named_modules():
            if isinstance(m, LoraInjectedLinear):
                # TODO check for requires_grad, currently assumes that if LoRA exists, it is being trained
                tensors_dict[f"{nc}.{nm}.lora_down.weight"] = m.lora_down.weight.cpu().clone()
                tensors_dict[f"{nc}.{nm}.lora_up.weight"] = m.lora_up.weight.cpu().clone()
                # Only non-diffusers modules will have metadata, which should contain
                # all the information necessary to reapply the to the pretrained model
                metadata[f"{nc}.{nm}:class"] = m.__class__.__name__
                metadata[f"{nc}.{nm}:r"] = str(m.r)
                metadata[f"{nc}.{nm}:scale"] = str(m.scale)
                metadata[f"{nc}.{nm}:nonlin"] = m.nonlin.__class__.__name__
            else:
                if nm=="":
                    pass
					# TODO some modules have no names, maybe moduleList?
                    #print("NO_MODULE_NAME", nm, type(m), "\t in child", nc, type(c))
                else:
                  for np, p in m.named_parameters():
                      if p.requires_grad:
                          #print(nm, type(m), "\t", np, type(p))
                          tensors_dict[f"{nc}.{nm}.{np}"] = p.cpu().clone()

    if validate:
        trainable_params = set()
        count = 0
        for n, p in model.named_parameters():
            count += 1
            if p.requires_grad:
                trainable_params.add(n)

        if tensors_dict.keys()==trainable_params:
            print("Copying trainable parameters:")
            print(f"\t {len(trainable_params)} trainable of {count} total parameters in have been copied to tensors_dict.")

            get_name = lambda k: k.split(":")[0]
            metadata_names = set()
            for k in metadata.keys():
                metadata_names.add(get_name(k))

            print(f"\t {len(metadata_names)} modules have associated metadata.")
        else:
            pass
            # TODO implement warning, print missing?

    return tensors_dict, metadata


def save_trainable_parameters(
    accelerator,
    tokenizer,
    text_encoder,
    unet,
    instance_token=None,
    instance_token_id=None,
    save_path,
#    dtype?
):
    to_save = {
    "embeddings": {},
    "text_encoder": {},
    "text_encoder_loras": {},
    "unet": {},
    "unet_loras": {},
    }

    # https://github.com/huggingface/diffusers/issues/1566
    # TODO move this up, rename extra_args > accelerator_unwrap_extra_args
    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
        inspect.signature(accelerator.unwrap_model).parameters.keys()
    )
    extra_args = (
        {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
    )

    if instance_token:
        # TODO: multi-token case
        token_embeddings = accelerator.unwrap_model(text_encoder, **extra_args).get_input_embeddings()
        instance_token_id = tokenizer.convert_tokens_to_ids(instance_token)
        trained_embeddings = token_embeddings.weight[instance_token_id]

        to_save["embeddings"][instance_token] = trained_embeddings.detach().cpu()
    if text_encoder:
        if accelerator:
            text_encoder_model = accelerator.unwrap_model(text_encoder)
        else:
            text_encoder_model = text_encoder

        trainable_dict = get_trainable_param_dict(text_encoder_model, **extra_args)
        to_save["text_encoder"] = trainable_dict["params_encoder"]
        to_save["text_encoder_loras"] = trainable_dict["params_loras"]
    if unet:
        if accelerator:
            unet_model = accelerator.unwrap_model(unet, **extra_args)
        else:
            unet_model = unet

        trainable_dict = get_trainable_param_dict(unet_model)
        to_save["unet"] = trainable_dict["params"]
        to_save["unet_loras"] = trainable_dict["params_loras"]

    torch.save(to_save, save_path)
    

def get_pipeline(
	model_name_or_path,
	vae_name_or_path=None, 
	tokenizer_name_or_path=None, 
	text_encoder_name_or_path=None,
	feature_extractor_name_or_path=None,
	scheduler=None,
	revision=None,
 	patch_dict,	
 	device="cuda",
):
	torch_dtype = torch.float16 if device == "cuda" else torch.float32

	if scheduler is None:
	    #scheduler = DPMSolverMultistepScheduler.from_pretrained(
	    #model_name_or_path, 
	    #subfolder="scheduler")

	    scheduler = DPMSolverMultistepScheduler(
	        beta_start=0.00085,
	        beta_end=0.012,
	        beta_schedule="scaled_linear",
	        num_train_timesteps=1000,
	        trained_betas=None,
	        prediction_type="epsilon",
	        thresholding=False,
	        algorithm_type="dpmsolver++",
	        solver_type="midpoint",
	        lower_order_final=True,
	    )

    pipe = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        custom_pipeline="lpw_stable_diffusion",
        safety_checker=None,
        revision=revision,
        scheduler=scheduler,
        vae=AutoencoderKL.from_pretrained(
            vae_name_or_path or model_name_or_path,
            subfolder=None if vae_name_or_path else "vae",
            revision=None if vae_name_or_path else revision,
            torch_dtype=torch_dtype,
        ),
        feature_extractor=feature_extractor_name_or_path,
        torch_dtype=torch_dtype
    )

    if patch_dict:
    	patch_pipeline(pipe, patch_dict)

    pipe.to(device)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe

# def patch_pipeline(
# 	pipeline,
# 	patch_dict,
# ):
# 	if patch_dict["embeddings"]:
# 		# model after load_learned_embed_in_clip
# 	if patch_dict["text_encoder"]:
# 		# use load_state_dict?
# 	if patch_dict["text_encoder_loras"]:
# 		# inject/patch Loras, model after monkeypatch_or_replace_lora, make replace_module_w_lora
# 	if patch_dict["unet"]:
# 		# use load_state_dict?
# 	if patch_dict["unet_loras"]:
# 		# inject/patch Loras, model after monkeypatch_or_replace_lora

# Function below is modified from
# MIT License

# Copyright (c) 2021 Gido M. van de Ven

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def count_parameters(model, verbose=True, file=sys.stdout):
    '''Count number of parameters, print to screen.'''
    total_params = trainable_params = fixed_params = 0
    for param in model.parameters():
        n_params = index_dims = 0
        for dim in param.size():
            n_params = dim if index_dims==0 else n_params*dim
            index_dims += 1
        total_params += n_params
        if param.requires_grad:
            trainable_params += n_params
        else:
            fixed_params += n_params
    if verbose:
        print("--> this network has {} parameters (~{} million)"
              .format(total_params, round(total_params / 1000000, 1)), file=file)
        print("      of which: - trainable: {} (~{} million)".format(trainable_params,
                                                                     round(trainable_params / 1000000, 1)), file=file)
        print("                - frozen: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)), file=file)
    return total_params, trainable_params, fixed_params
