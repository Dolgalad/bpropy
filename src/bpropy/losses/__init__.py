from .registry import LOSS_REGISTRY, register_loss

from .ce import (
        loss_bce_logits,
        loss_factor_ce_from_logits_factor_sizes,
        loss_factor_focal_ce_from_logits_factor_sizes,
        loss_factor_ce_from_marginals_factor_sizes,
        loss_factor_focal_ce_from_marginals_factor_sizes
        )

from .bethe import loss_nll_bethe, loss_nll_bethe_from_logits
from .local_consistency import loss_local_consistency_l2, loss_local_consistency_kl
from .nll_grad_only import loss_nll_grad_only



