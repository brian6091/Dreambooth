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

from safetensors.torch import save_file as safe_save
from safetensors import safe_open
    

def get_tensor_info(tensor):
    info = []
    for name in ['is_leaf', 'requires_grad', 'retains_grad', 'grad_fn', 'grad']:
        info.append(f'{name}({getattr(tensor, name, None)})')
    return ' '.join(info)


def print_trainable_parameters(model: nn.Module, file=sys.stdout, tensor_info=True):
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape, file=file)
            if tensor_info:
                print(get_tensor_info(p), file=file)


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
    tokenizer,
    text_encoder,
    unet,
    instance_token=None,
    save_path="./lora.safetensors",
#    dtype?
):
    md = {
        "version": "__0.1.0__",
        "separator": ":",
        "token_embedding_prefix": "token_embedding",
        "text_encoder_prefix": "text_encoder",
        "unet_prefix": "unet",
    }
    td_token_embedding = {}
    md_token_embedding = {}
    td_text_encoder = {}
    md_text_encoder = {}
    td_unet = {}
    md_unet = {}

    if instance_token:
        # TODO: multi-token case
        token_embeddings = text_encoder.get_input_embeddings()
        instance_token_id = tokenizer.convert_tokens_to_ids(instance_token)
        trained_embeddings = token_embeddings.weight[instance_token_id]

        k = f"token_embedding:{instance_token}"
        td_token_embedding[k] = trained_embeddings.detach().cpu()
        md_token_embedding[k] = str(instance_token_id)
    if text_encoder:
        td_text_encoder, md_text_encoder = get_trainable_param_dict(text_encoder)
        # PREFIX keys
    if unet:
        td_unet, md_unet = get_trainable_param_dict(unet)
        # PREFIX keys

    tensors_dict = {**td_token_embedding, **td_text_encoder, **td_unet}
    metadata = {**md, **md_token_embedding, **md_text_encoder, **md_unet}

    print(f"Saving weights to {save_path}")
    safe_save(tensors_dict, save_path, metadata)
    

def load_trained_parameters(
	filename,
    framework="pt",
    device="cpu",
):
    metadata = {}
    tensors_dict_loaded = {}
    with safe_open(filname, framework=framework, device=device) as f:
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
