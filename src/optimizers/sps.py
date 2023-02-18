# BSD 3-Clause License
# Copyright (c) 2023, fabian-sp
# https://github.com/fabian-sp/ProxSPS

import torch
import warnings

class SPS(torch.optim.Optimizer):
    def __init__(self, params, lr: float=1e-3, weight_decay: float=0, fstar: float=0, prox: bool=True):
        """
        
        Parameters
        ----------
        params : 
            PyTorch model parameters.
        lr : float, optional
            Learning rate. The default is 1e-3.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        fstar : float, optional
            Lower bound of loss function. The default is 0 (which is a lower bound for most loss functions).

        prox: bool, optional
            Whether to use ProxSPS or SPS.
            
        """
        
        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        
        super(SPS, self).__init__(params, defaults)
        self.params = params
        
        self.lr = lr
        self.fstar = fstar
        self.prox = prox

        self.state['step_size_list'] = list()
        
        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")
        
        return
        
    def step(self, closure=None, loss=None):
        """
        Step for (Prox)SPS method.

        Parameters
        ----------
        closure : Callable, optional
            Function that computes the loss function value. Either this or loss must be specified.
            The default is None.
        loss : optional
            Loss function value. The default is None.
            
               
        Returns
        -------
        Loss function value
            As float.

        """
        if len(self.param_groups) > 1:
            warnings.warn("Multiple parameter groups in SPS!")
        
        if loss is None and closure is None:
            raise ValueError('please specify either closure or loss')

        if loss is not None:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss)
                
        # get fstar
        fstar = self.fstar
        
        ############################################################
        # compute loss and gradients
        if loss is None:
            loss = closure()
        else:
            assert closure is None, 'If ``loss`` is provided then ``closure`` should be None'

        # add l2-norm if not ProxSPS
        if not self.prox:
            r = 0
            
            for group in self.param_groups:
                lmbda = group['weight_decay']
                for p in group['params']:
                    p.grad.add_(lmbda * p.data)  # gradients
                    r += (lmbda/2) * (p.data**2).sum() # loss
                    
            loss.add_(r) 
        
                
        if self.prox:
            grad_norm, grad_dot_w = self.compute_grad_terms(need_gdotw=True)
        else:
            grad_norm, grad_dot_w = self.compute_grad_terms(need_gdotw=False)
            assert grad_dot_w == 0.
        
        ############################################################
        # update 
        for group in self.param_groups:
            lr = group['lr']
            lmbda = group['weight_decay']
            
            for p in group['params']:
                if self.prox:
                    nom = (1+lr*lmbda)*(loss - fstar) - lr*lmbda*grad_dot_w
                else:
                    nom = loss - fstar
                    
                denom = (grad_norm)**2 
                t1 = (nom/denom).item()
                t2 = max(0., t1)                 
                if not self.prox:
                    assert t1 >= 0 # always t2 = t1 for SPS
        
                # compute tau^+
                tau = min(lr, t2) 
                
                p.data.add_(other=p.grad, alpha=-tau)
                if self.prox:
                    p.data.div_(1+lr*lmbda)
            
        ############################################################
        
        # update state with metrics
        self.state['step_size_list'].append(t2) # works only if one param_group!

        return float(loss)
    
        
    def compute_grad_terms(self, need_gdotw=True):
        """
        computes:
            - norm of stochastic gradient ||g||
            - inner product <g,w> where w is current weights (optional). 
        """
        grad_norm = 0.
        grad_dot_w = 0.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    raise KeyError("None gradient")
                
                g = p.grad
                grad_norm += torch.sum(torch.mul(g, g))
                if need_gdotw:
                    grad_dot_w += torch.sum(torch.mul(p, g))
          
        grad_norm = torch.sqrt(grad_norm)
        return grad_norm, grad_dot_w
