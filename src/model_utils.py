from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union
import torch.nn as nn
from lora_diffusion import LoraInjectedLinear


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    for param in params:
        param.requires_grad = True


def print_trainable_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


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


def _inject_trainable_lora(
    model: nn.Module,
    target_name: str,
    r: int = 4,
    train_off_target: Set[str] = None,
):
    """
    inject lora for nn.Linear into model.
    """

    if target_name==None:
        for p, n, m in _find_children(model):
            if not isinstance(p, LoraInjectedLinear):
                _inject_trainable_lora(p, target_name=n, r=r)
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
                r,
            )

            # Assign pretrained parameters
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias

            # Switch the module
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


def set_trainable_parameters(
    model: nn.Module,
    target_module_or_class: Set[str],
    target_submodule: Set[str],
    lora_layer: Set[str] = None,
    lora_train_off_target: Set[str] = None,
    lora_rank: int = None,
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
                                train_off_target=lora_train_off_target,
                                )

                            
# Functions below here modified from
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

def count_parameters(model, verbose=True):
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
              .format(total_params, round(total_params / 1000000, 1)))
        print("      of which: - learnable: {} (~{} million)".format(trainable_params,
                                                                     round(trainable_params / 1000000, 1)))
        print("                - fixed: {} (~{} million)".format(fixed_params, round(fixed_params / 1000000, 1)))
    return total_params, trainable_params, fixed_params


def print_model_info(model, title="MODEL"):
    '''Print information on [model] onto the screen.'''
    print("\n" + 40*"-" + title + 40*"-")
    print(model)
    print(90*"-")
    _ = count_parameters(model)
    print(90*"-")
