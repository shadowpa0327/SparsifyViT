import torch.nn as nn
import torch.nn.functional as F
import torch

def compute_mask(t, N, M):
    out_channel, in_channel = t.shape
    percentile = N / M
    t_reshaped = t.reshape(out_channel, -1, M)

    mask = torch.ones_like(t)
    mask_reshaped = mask.reshape(out_channel, -1, M)
    
    nparams_topprune = int(M * (1-percentile)) 
    if nparams_topprune != 0:
        topk = torch.topk(torch.abs(t_reshaped), k=nparams_topprune, largest=False, dim = -1)
        mask_reshaped = mask_reshaped.scatter(dim = -1, index = topk.indices, value = 0)
    
    return mask_reshaped.reshape(out_channel, in_channel)

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta 
    
    def __repr__(self):
        return f"Affine(dim={self.dim})"

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
        
        self.sparsity_idx = 0
        self.sparsity_config = (4, 4)
        self.nas_config_list = ['no_separate']
        self.mask = torch.ones_like(self.weight)
        self.set_sample_config(self.sparsity_config)
        self.per_candidate_affine_modules = None
        
        
    def set_seperate_config(self, seperate_configs): 
        # supernet training: used after loading pre-trained weights; ea: used before loading nas weights
        if seperate_configs:
            self.nas_config_list = seperate_configs
            # Repeat
            self.weight = nn.Parameter(self.weight.repeat(len(self.nas_config_list), 1, 1))
            if self.bias != None:
                self.bias = nn.Parameter(self.bias.repeat(len(self.nas_config_list), 1))

    def set_indeped_affine(self, candidate_configs):
        module_dict = {}
        for sparsity_cfg in candidate_configs:
            assert isinstance(sparsity_cfg, (tuple, list)) and len(sparsity_cfg) == 2
            n, m = sparsity_cfg
            module_dict[f"n{n}_m{m}"] = Affine(self.out_features)
        
        self.per_candidate_affine_modules = nn.ModuleDict(module_dict)
        
    
    def set_sample_config(self, sample_config):
        self.sparsity_config = sample_config
        self._set_mask()
        
    def _set_mask(self):
        n, m = self.sparsity_config
        # Find the corresponding index
        if len(self.nas_config_list) == 1: # No separate
            self.mask = compute_mask(self.weight, n, m)
        else:
            for config in self.nas_config_list:
                if [n, m] in config:
                    self.sparsity_idx = self.nas_config_list.index(config)
            self.mask = compute_mask(self.weight[self.sparsity_idx], n, m)

    def __repr__(self):
        info = f"SparseLinearSuper(in_features={self.in_features}, out_features={self.out_features}, sparse_config:{self.sparsity_config}, \
per_choices_affine:{self.per_candidate_affine_modules is not None})"
        if self.per_candidate_affine_modules is not None:
            for name, affine_module in self.per_candidate_affine_modules.items():
                info+=f"\n  ({name}): {affine_module.__repr__()}"
        return info
    
    def forward(self, x):
        if len(self.nas_config_list) == 1: # No seperate
            weight = self.weight * self.mask
        else:  # separate
            weight = self.weight[self.sparsity_idx] * self.mask

        if self.bias is not None:
            if len(self.nas_config_list) == 1: # No seperate
                x = F.linear(x, weight, self.bias)
            else:  # seperate
                x = F.linear(x, weight, self.bias[self.sparsity_idx])
        else:
            x = F.linear(x, weight)
        
        if self.per_candidate_affine_modules is not None:
            cur_n, cur_m = self.sparsity_config
            x = self.per_candidate_affine_modules[f"n{cur_n}_m{cur_m}"](x)
        
        return x
    
    def get_sparse_level(self):
        n, m = self.sparsity_config
        return (n / m)
    
    def num_pruned_params(self):
        if self.mask.size() == self.weight.size():
            return int(torch.sum(self.mask==0).item())
        else:
            return int(torch.sum(self.mask==0).item()) + self.weight[0].numel() * (len(self.nas_config_list) - 1)




if __name__ == '__main__':
    m = SparseLinearSuper(12, 12)
    input = torch.randn(12)
    print(m(input))
    m.set_seperate_config(seperate_configs=[[[1, 4], [2, 4]], [[4, 4]]])
    m.set_sample_config((1, 4))
    print(m(input))
    print(m.num_pruned_params())
    #print(sum(p.numel() for p in m.parameters() if p.requires_grad))
    
