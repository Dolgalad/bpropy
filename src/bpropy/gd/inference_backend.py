from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, Tuple

Evidence = Dict[int, int]  # var -> {0,1}

class InferenceBackend(Protocol):
    def reset(self, pb) -> None:
        """Initialize backend for a new problem instance."""

    def observe(self) -> Optional[Any]:
        """
        Return whatever the adapter may use:
          - for inference: mu, beliefs, diagnostics, etc.
          - for heuristics: scores, features, etc.
          - can be None if not needed
        """

    def update(self, evidence: Evidence) -> None:
        """Incorporate new evidence / decisions (clamp vars, update state)."""

