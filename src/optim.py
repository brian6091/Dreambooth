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
        
        opts['AdaBound'] = optim.AdaBound
        opts['Adahessian'] = optim.Adahessian
        opts['AdamP'] = optim.AdamP
        opts['DiffGrad'] = optim.DiffGrad
        opts['MADGRAD'] = optim.MADGRAD
        opts['QHAdam'] = optim.QHAdam
        opts['Yogi'] = optim.Yogi
    except:
        pass

    opts = opts | {'Adagrad': torch.optim.Adagrad, 'Adam': torch.optim.Adam, 
                   'AdamW': torch.optim.AdamW, 'RAdam': torch.optim.RAdam, 
                   'SGD': torch.optim.SGD}
    
    if optname in opts:
        optimizer_class = opts[optname]
    else:
        raise ValueError(
            f"Optimizer {optname} not supported yet."
        )
        
    return optimizer_class
