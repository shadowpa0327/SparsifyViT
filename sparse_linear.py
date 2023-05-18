import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch

class Sparse(autograd.Function): # from SR-STE
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, decay = 0.0002):
        ctx.save_for_backward(weight)

        output = weight.clone()
        length = weight.numel()
        group = int(length/M)

        weight_temp = weight.detach().abs().reshape(group, M)
        index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

        w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
        ctx.mask = w_b
        ctx.decay = decay

        return output*w_b


    @staticmethod
    def backward(ctx, grad_output):

        weight, = ctx.saved_tensors
        return grad_output + ctx.decay * (1-ctx.mask) * weight, None, None

def compute_mask(t, N, M):
    out_channel, in_channel = t.shape
    percentile = N / M
    t_reshaped = t.reshape(out_channel, -1, M)
    #print(t_reshaped.shape)
    mask = torch.ones_like(t)
    mask_reshaped = mask.reshape(out_channel, -1, M)
    
    nparams_topprune = int(M * (1-percentile)) 
    if nparams_topprune != 0:
        topk = torch.topk(torch.abs(t_reshaped), k=nparams_topprune, largest=False, dim = -1)
        mask_reshaped = mask_reshaped.scatter(dim = -1, index = topk.indices, value = 0)
    
    return mask_reshaped.reshape(out_channel, in_channel)

class SparseLinearSuper(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.ones(out_features))
        else:
            self.bias = None

        self.sparsity_config = (4, 4)
        self.mask = torch.ones_like(self.weight)
        self.set_sample_config(self.sparsity_config)

    def set_sample_config(self, sample_config):
        self.sparsity_config = sample_config
        self._set_mask()
        
    def _set_mask(self):
        n, m = self.sparsity_config
        self.mask = compute_mask(self.weight, n, m)

    def __repr__(self):
        return f"SparseLinearSuper(in_features={self.in_features}, out_features={self.out_features}, sparse_config:{self.sparsity_config})"
    
    def get_sparse_weights(self):
        n, m = self.sparsity_config
        return Sparse.apply(self.weight, n, m)

    def forward(self, x):
        weight = self.get_sparse_weights()
        #weight = self.weight * self.mask
        #weight = self.weight
        if self.bias is not None:
            x = F.linear(x, weight, self.bias)
        else:
            x = F.linear(x, weight)

        return x
    
    def num_pruned_params(self):
        return int(torch.sum(self.mask==0).item())


if __name__ == '__main__':
    m = SparseLinearSuper(12, 12)
    input = torch.randn(12)
    print(m(input))
    m.set_sample_config((1,4))
    print(m(input))
    print(m.num_pruned_params())
    #print(sum(p.numel() for p in m.parameters() if p.requires_grad))
    