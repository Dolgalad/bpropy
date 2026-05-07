import torch

def mean_embedding_distance(h):
    ## Mean embedding distance
    dists = torch.pdist(h, p=2)
    return dists.mean()
    embed_dist = 0.0
    for i in range(h.shape[0]):
        for j in range(h.shape[0]):
            if i<j:
                embed_dist += torch.linalg.norm( h[i,:] - h[j,:])
    embed_dist *= 2 / (h.shape[0] * (h.shape[0] - 1))
    return embed_dist

def mean_embedding_std(h):
    ## Embedding variance
    embed_std = torch.std(h, dim=0).mean()
    return embed_std

def spectral_effective_rank(h):
    ## Spectrum
    U,S,Vh = torch.linalg.svd(h)
    p = S / S.sum()
    entropy_p = -(p * torch.log(p + 1e-12)).sum()
    effective_rank = torch.exp(entropy_p)
    return effective_rank

def condition_number(h):
    U,S,Vh = torch.linalg.svd(h)
    condition_number = S.max() / S.min()
    return condition_number
