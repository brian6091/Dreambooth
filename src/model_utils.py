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

import torch
import torch.nn as nn
from diffusers import (
    DPMSolverMultistepScheduler,
    DDIMScheduler, 
    DDPMScheduler, 
    LMSDiscreteScheduler, 
    PNDMScheduler, 
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
         
from lora_diffusion import LoraInjectedLinear

from safetensors.torch import save_file as safe_save
from safetensors import safe_open


# TODO make frozen, probably need another dict for how lora is saved in metadata?
SAFE_CONFIGS = {
    "0.0.0": { # Reserved for lora library
        "version": "__0.0.0__",
        "separator": ":",
        "token_embedding_prefix": "",
        "text_encoder_prefix": "",
        "unet_prefix": "",
        "lora_prefix": "",
        "token_is_key": "True",
    },
    "0.1.0": {
        "version": "__0.1.0__",
        "separator": ":",
        "token_embedding_prefix": "token_embedding",
        "text_encoder_prefix": "text_encoder",
        "unet_prefix": "unet",
        "lora_prefix": "lora",
        "lora_weight_names": {'lora_down', 'lora_up'},
    }
}


def get_tensor_info(tensor):
    info = []
    for name in ['is_leaf', 'requires_grad', 'retains_grad', 'grad_fn', 'grad']:
        info.append(f'{name}({getattr(tensor, name, None)})')
    return ' '.join(info)


def print_trainable_parameters(model: nn.Module, file=sys.stdout, tensor_info=False):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape, file=file)
            if tensor_info:
                print(get_tensor_info(p), file=file)


def get_noise_scheduler(
	scheduler: str,
	config=None,
    model_name_or_path=None,
):
    if scheduler=="DPMSolverMultistepScheduler":
        if model_name_or_path:
            noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = DPMSolverMultistepScheduler.from_config(config if config else {})
    elif scheduler=="DDIMScheduler":
        if model_name_or_path:
            noise_scheduler = DDIMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = DDIMScheduler.from_config(config if config else {})
    elif scheduler=="DDPMScheduler":
        if model_name_or_path:
            noise_scheduler = DDPMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = DDPMScheduler.from_config(config if config else {})
    elif scheduler=="LMSDiscreteScheduler":
        if model_name_or_path:
            noise_scheduler = LMSDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = LMSDiscreteScheduler.from_config(config if config else {})
    elif scheduler=="PNDMScheduler":
        if model_name_or_path:
            noise_scheduler = PNDMScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = PNDMScheduler.from_config(config if config else {})
    elif scheduler=="EulerDiscreteScheduler":
        if model_name_or_path:
            noise_scheduler = EulerDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = EulerDiscreteScheduler.from_config(config if config else {})
    elif scheduler=="EulerAncestralDiscreteScheduler":
        if model_name_or_path:
            noise_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_name_or_path, subfolder="scheduler")
        else:
            noise_scheduler = EulerAncestralDiscreteScheduler.from_config(config if config else {})
    else:
        raise ValueError(
            f"Unknown scheduler: {scheduler}"
        )        
	
    return noise_scheduler
	

def add_instance_tokens(
    tokenizer,
    text_encoder,
    instance_token,
    initializer_token=None,
    embedding=None,
    debug=False,
):
    # TODO: multiple tokens
    num_added_tokens = tokenizer.add_tokens(instance_token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {instance_token}. Please pass a different"
            " `instance_token` that is not already in the tokenizer."
        )
    else:
        if debug:
            print(f"{instance_token} added to tokenizer.")

    # Resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    token_embeds = text_encoder.get_input_embeddings().weight.data
    instance_token_id = tokenizer.convert_tokens_to_ids(instance_token)
    
    initializer_token_id = None
    if embedding is not None:
        # TODO check tensor is right format
        token_embeds[instance_token_id] = embedding
        
        if initializer_token:
            print(f"Initializer tokens {intializer_token} ignored since an embedding was provided")
    elif initializer_token is not None:
        # Initialise new instance_token embedding with the embedding of the initializer_token
        initializer_token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
	
        if len(initializer_token_ids) > 1:
            # Take the mean, does this mean anything?
            initial_embed = torch.mean(token_embeds[initializer_token_ids,], 0)
        else:
            initial_embed = token_embeds[initializer_token_id]
#         initializer_token_id = token_ids[0]
#         if len(token_ids) > 1:
#             raise ValueError("The initializer token must be a single token.")

        if debug:
            print("Instance weights: ")
            print(token_embeds[instance_token_id])

        token_embeds[instance_token_id] = initial_embed

        if debug:
            print("Instance weights intialized: ")
            print(token_embeds[instance_token_id])
    else:
        print(f"Embedding vector for {instance_token} is random.")
        #pass
        # TODO if no initializer_token, initialize to zero?

    return instance_token_id, initializer_token_id


def find_modules_by_name_or_class(
    model: nn.Module,
    target: Set[str],
):
    for fullname, module in model.named_modules():
        *path, name = fullname.split(".")
        if (module.__class__.__name__ in target) or (name in target):
            yield module.__class__.__name__, fullname, name, module
            

# TODO import
# This function is from: https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# SPDX short identifier: Apache-2.0
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


# This function adapted from: https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
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


def _inject_trained_lora(
    module: nn.Module,
    target: str,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    r: int = 4,
    scale: float = 1.0,
    nonlin: nn.Module = None,
):
    """
    inject lora for nn.Linear into model.
    """

    # TODO return if not in replacable class
    if not isinstance(module._modules[target], LoraInjectedLinear):
        _child_module = module._modules[target]
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
        
        # Assign pretrained parameters
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # Switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        module._modules[target] = _tmp

        module._modules[target].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        module._modules[target].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )
    else:
        # TODO if already LoRAInjected? Replace
        print(f"skipping {target}")


def get_modules_to_inject_with_parent(
    model: nn.Module,
    lora_modules,
):
    # Parent module names
    parent_modules = [n.rsplit('.', 1)[0] for n in lora_modules]

    for n, m in model.named_modules():
        if any(n==c for c in parent_modules):
            # Some elements (ModuleLists) will get traversed twice, e.g., attn1, and attn1.to_out
            # since to_out is a list of modules. They will get skipped in the injection, so it's
            # just a little wasteful, but no harm done...
            for _n, _m in m.named_modules():
                if any(f"{n}.{_n}"==c for c in lora_modules):
                    # Handle case where child is a ModuleList
                    if "." in _n:
                        # The name should have two parts
                        parts = _n.split(".")
                        # Only know how to handle this case
                        if (len(parts)) > 2:
                            print("unexpected???")
                        if not isinstance(m._modules["to_out"], nn.ModuleList):
                            print("unexpected???")
                        # Reassign the parent 
                        m = m._modules[parts[0]]
                        n = f"{n}.{parts[0]}"
                        # Reassign the child
                        _m = m._modules[parts[1]]
                        _n = parts[1]

                    yield n, m, _n, _m


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
    exclude_params: Set[str] = {},
    validate=True,
    config=SAFE_CONFIGS["0.1.0"],
    dtype=torch.float32,
):
    cf = config.copy()
    tensors_dict = {}
    metadata = {}

    exclude_params = {"weight", *exclude_params}

    saved = []
    for nc, c in model.named_children():
        for nm, m in c.named_modules():
            if isinstance(m, LoraInjectedLinear):
                # Only non-diffusers modules will have metadata, which should contain
                # all the information necessary to reapply to the pretrained model
                prefix = f"{cf['lora_prefix']}{cf['separator']}{nc}.{nm}"       
                metadata[f"{prefix}{cf['separator']}class"] = m.__class__.__name__
                metadata[f"{prefix}{cf['separator']}r"] = str(m.r)
                metadata[f"{prefix}{cf['separator']}scale"] = str(m.scale)
                metadata[f"{prefix}{cf['separator']}nonlin"] = m.nonlin.__class__.__name__

            if nm=="":
                pass
                # TODO some modules don't have names, maybe moduleList?
                if validate:
                    print("\tNO_NAME for module", nm, type(m), "\n\t\t child of ", nc, type(c))
            else:
                for np, p in m.named_parameters():
                    if p.requires_grad and (np in exclude_params):
                        print(f"\n Requires_grad True for {np} in module {nm}, but not saved by request.\n")
                    if p.requires_grad and (np not in exclude_params):
                        #print(nm, type(m), "\t", np, type(p))
                        if any(x in np for x in cf["lora_weight_names"]):
                            k = f"{cf['lora_prefix']}{cf['separator']}{nc}.{nm}.{np}"
                        else:
                            k = f"{nc}.{nm}.{np}"

                        if k not in saved:
                            print(f"\t saving with key: {k}")
                            saved.append(k)
                            tensors_dict[k] = p.cpu().clone()
                            #tensors_dict[k] = p.cpu().clone().to(dtype)
                        
    if validate:
        trainable_params = set()
        count = 0
        for n, p in model.named_parameters():
            count += 1
            if p.requires_grad:
                trainable_params.add(n)

        if tensors_dict.keys()==trainable_params:
            print("Copying trainable parameters:")
            print(f"\t {len(trainable_params)} trainable of {count} total parameters copied to tensors_dict.")

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
    tokenizer,
    text_encoder,
    unet,
    instance_token=None,
    save_path="./lora.safetensors",
    config=SAFE_CONFIGS["0.1.0"],
#    dtype?
):
    cf = config.copy()
    td_token = {}
    md_token = {}
    td_text = {}
    md_text = {}
    td_unet = {}
    md_unet = {}

    if instance_token:
        # TODO: multi-token case
        token_embeddings = text_encoder.get_input_embeddings()
        instance_token_id = tokenizer.convert_tokens_to_ids(instance_token)
        trained_embeddings = token_embeddings.weight[instance_token_id]

        k = f"{cf['token_embedding_prefix']}{cf['separator']}{instance_token}"
        td_token[k] = trained_embeddings.detach().cpu()
        md_token[k] = str(instance_token_id)
    if text_encoder:
        if instance_token:
            # instance_token added to tokenizer, but all the other embeds are frozen.
            # The embedding will thus have requires_grad=TRUE, but we do not want to save it
#             td_text, md_text = get_trainable_param_dict(text_encoder, exclude_params = {"token_embedding.weight"})
            td_text, md_text = get_trainable_param_dict(text_encoder)
        else:
            td_text, md_text = get_trainable_param_dict(text_encoder)
            
        td_text = {f"{cf['text_encoder_prefix']}{cf['separator']}{k}": v for k, v in td_text.items()}
        md_text = {f"{cf['text_encoder_prefix']}{cf['separator']}{k}": v for k, v in md_text.items()}
    if unet:
        td_unet, md_unet = get_trainable_param_dict(unet)
        td_unet = {f"{cf['unet_prefix']}{cf['separator']}{k}": v for k, v in td_unet.items()}
        md_unet = {f"{cf['unet_prefix']}{cf['separator']}{k}": v for k, v in md_unet.items()}

    tensors_dict = {**td_token, **td_text, **td_unet}
    # Safetensors requires metadata to be flat and text only
#     if cf["version"]=="0.1.0":
#         # TODO, this should generally apply to any config to make it safetensors compatible
#         cf["lora_weight_names"] = str(cf["lora_weight_names"])
    del cf["lora_weight_names"]
    metadata = {**cf, **md_token, **md_text, **md_unet}

    print(f"Saving weights with format version {cf['version']} to {save_path}")
    safe_save(tensors_dict, save_path, metadata)
    
    return [k for k in tensors_dict.keys()]
    

def load_trained_parameters(
    filename,
    framework="pt",
    device="cpu",
):
    metadata = {}
    tensors_dict = {}
    with safe_open(filename, framework=framework, device=device) as f:
        metadata = f.metadata()
        for k in f.keys():
            tensors_dict[k] = f.get_tensor(k)
            
    return tensors_dict, metadata
	
	
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
