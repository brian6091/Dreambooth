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
from typing import Tuple, Union

import math

import torch
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

def calculate_loss(input, target, loss_function="mse", reduction="mean", beta=1.0):
    if loss_function in ("mse", "MSE"):
        loss = F.mse_loss(input, target, reduction=reduction)
    elif loss_function in ("l1", "L1"):
        loss = F.l1_loss(input, target, reduction=reduction)
    elif loss_function in ("smoothl1", "smoothL1"):
        loss = F.smooth_l1_loss(input, target, beta=beta, reduction=reduction)        

    return loss


def load_optimizer(optname):    
    opts = dict()
    try:
        import bitsandbytes as bnb
        
        opts['AdamW8bit'] = bnb.optim.AdamW8bit
    except:
        pass

    try:
        import torch_optimizer as optim
        
        # https://github.com/jettify/pytorch-optimizer
        opts['AdaBound'] = optim.AdaBound
        opts['AdamP'] = optim.AdamP
        opts['DiffGrad'] = optim.DiffGrad
        opts['MADGRAD'] = optim.MADGRAD
        opts['QHAdam'] = optim.QHAdam
        opts['Yogi'] = optim.Yogi
    except:
        pass

    opts2 = {'Adagrad': torch.optim.Adagrad, 'Adam': torch.optim.Adam, 
            'AdamW': torch.optim.AdamW, 'RAdam': torch.optim.RAdam, 
            'NAdam': torch.optim.NAdam, 'Adamax': torch.optim.Adamax,
             'SGD': torch.optim.SGD}
    
    opts = {**opts, **opts2}
    
    if optname in opts:
        optimizer_class = opts[optname]
    else:
        raise ValueError(
            f"Optimizer {optname} not supported yet."
        )
        
    return optimizer_class


def group_parameters(unet,
                     lr_unet,
                     text_encoder,
                     lr_text,
                     lr_scaling=1.0,
                     separate_token_embedding=False, 
                     lr_token_embedding=None,
                     debug=False,
):
    unet_params_to_optimize = {
        "name": "unet",
        "params": [p for p in unet.parameters() if p.requires_grad],
        "lr": lr_unet*lr_scaling,
    }
    unet_params_to_optimize["n_params"] = len(unet_params_to_optimize["params"])
    train_unet = unet_params_to_optimize["n_params"]>0

    text_token_embedding = []
    text_nontoken = []
    count = 0
    for n, p in text_encoder.named_parameters():
        if p.requires_grad:
            count += 1
            if n.find("token_embedding")>0:
                text_token_embedding.append(p)
            else:
                text_nontoken.append(p)
                
    if separate_token_embedding:                    
        count2 = 0
        for p in text_encoder.parameters():
            if p.requires_grad:
                count2 += 1
                
        print(f"{count2} parameters set to be trained. Found {count}, with {len(text_token_embedding)} token embeddings, and {len(text_nontoken)} others in text encoder")
        
        # Token embedding alone
        token_embedding_to_optimize = {
            "name": "token_embedding",
            "params": text_token_embedding,
            "lr": lr_token_embedding*lr_scaling,
        }
        token_embedding_to_optimize["n_params"] = len(text_token_embedding)
        train_token_embedding = len(text_token_embedding)>0

        # Everything else goes into another group
        text_params_to_optimize = {
            "name": "text_encoder",
            "params": text_nontoken,
            "lr": lr_text*lr_scaling,
        }
        text_params_to_optimize["n_params"] = len(text_nontoken)
        train_text_encoder = len(text_nontoken)>0
    else:
        # Group all of text_encoder together
        text_params_to_optimize = {
            "name": "text_encoder",
            "params": [p for p in text_encoder.parameters() if p.requires_grad],
            "lr": lr_text*lr_scaling,
        }
        text_params_to_optimize["n_params"] = len(text_params_to_optimize["params"])
        train_token_embedding = len(text_token_embedding)>0
        train_text_encoder = len(text_nontoken)>0
    
    params_to_optimize = []
    if train_token_embedding and separate_token_embedding:
        params_to_optimize.append(token_embedding_to_optimize)
    if train_text_encoder:
        params_to_optimize.append(text_params_to_optimize)    
    if train_unet:
        params_to_optimize.append(unet_params_to_optimize)
        
    return train_token_embedding, train_text_encoder, train_unet, params_to_optimize
    
    
def get_explore_exploit_schedule_with_warmup(
    optimizer: Optimizer,
    start_step: Union[Tuple[int], int],
    num_warmup_steps: Union[Tuple[int], int],
    num_explore_steps: Union[Tuple[int], int],
    num_total_steps: Union[Tuple[int], int],
    last_epoch: int = -1
):
    """
    Explore-Exploit learning rate schedule (Knee schedule)
    https://arxiv.org/pdf/2003.03977.pdf
    
    Setting num_explore_steps=0 will reproduce Slanted triangular learning rates
    https://arxiv.org/pdf/1801.06146.pdf
    """

    #TODO: assert all ints or all Tuple[int] of same length

    def factory(start, warmup, explore, total):
        def f(current_step):
            if current_step <= start:
                return 0.0
            if current_step <= (warmup + start):
                return float(current_step - start) / float(max(1, warmup))
            elif current_step <= (explore + warmup + start):
                return 1.0
            else:
                return max(
                    0.0, float(total - current_step) / float(max(1, total - warmup - explore - start))
                )

        return f

    if isinstance(start_step, int):
#         def lr_lambda(current_step):
#             if current_step <= start_step:
#                 return 0.0
#             if current_step <= (num_warmup_steps + start_step):
#                 return float(current_step - start_step) / float(max(1, num_warmup_steps))
#             elif current_step <= (num_explore_steps + num_warmup_steps + start_step):
#                 return 1.0
#             else:
#                 return max(
#                     0.0, float(num_total_steps - current_step) / float(max(1, num_total_steps - num_warmup_steps - num_explore_steps - start_step))
#                 )
        lr_lambda = factory(start_step, num_warmup_steps, num_explore_steps, num_explore_steps)

#         return LambdaLR(optimizer, lr_lambda, last_epoch)
    else:
        lr_lambda = []
        for start, warmup, explore, total in zip(start_step, num_warmup_steps, num_explore_steps, num_total_steps):
            lr_lambda.extend([factory(start, warmup, explore, total)])

    return LambdaLR(optimizer, lr_lambda, last_epoch)
