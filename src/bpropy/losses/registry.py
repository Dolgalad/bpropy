from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple


LOSS_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    #"bce_logits": loss_bce_logits,
    #"fce_logits": loss_factor_ce_from_logits_factor_sizes,
    #"ffce_logits": loss_factor_focal_ce_from_logits_factor_sizes,
    #"fce": loss_factor_ce_from_marginals_factor_sizes,
    #"ffce": loss_factor_focal_ce_from_marginals_factor_sizes,
    #"nll_bethe": loss_nll_bethe,
    #"nll_bethe_logits": loss_nll_bethe_from_logits,
    #"local_consistency_l2": loss_local_consistency_l2,
    #"local_consistency_kl": loss_local_consistency_kl,

    # Later: "factor_ce", "focal_ce", etc.
}

def register_loss(name):
    def deco(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return deco

