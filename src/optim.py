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
import torch

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
    train_unet = len(unet_params_to_optimize["params"])>0

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
        # Put token_embedding into it's own parameter group
#         text_token_embedding = []
#         text_nontoken = []
#         count = 0
#         for n, p in text_encoder.named_parameters():
#             if p.requires_grad:
#                 count += 1
#                 if n.find("token_embedding")>0:
#                     text_token_embedding.append(p)
#                 else:
#                     text_nontoken.append(p)
                    
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
        train_token_embedding = len(text_token_embedding)>0

        # Everything else goes into another group
        text_params_to_optimize = {
            "name": "text_encoder",
            "params": text_nontoken,
            "lr": lr_text*lr_scaling,
        }
        train_text_encoder = len(text_nontoken)>0
    else:
        # Group all of text_encoder together
        text_params_to_optimize = {
            "name": "text_encoder",
            "params": [p for p in text_encoder.parameters() if p.requires_grad],
            "lr": lr_text*lr_scaling,
        }
        
        train_token_embedding = len(text_token_embedding)>0
#         train_token_embedding = False # Note that embedding may be trained, but not in separate group
#         for n, p in text_encoder.named_parameters():
#             if p.requires_grad and (n.find("token_embedding")>0):
#                 train_token_embedding = True
         
        train_text_encoder = len(text_nontoken)>0
        #train_text_encoder = len(text_params_to_optimize["params"])>0
    
    params_to_optimize = []
    if train_token_embedding and separate_token_embedding:
        params_to_optimize.append(token_embedding_to_optimize)
    if train_text_encoder:
        params_to_optimize.append(text_params_to_optimize)    
    if train_unet:
        params_to_optimize.append(unet_params_to_optimize)
        
    return train_token_embedding, train_text_encoder, train_unet, params_to_optimize
    
