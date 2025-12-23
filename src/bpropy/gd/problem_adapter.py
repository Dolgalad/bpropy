from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, Tuple

class ProblemAdapter(Protocol):
    def reset(self, pb) -> None:
        """Initialize adapter state for new problem."""

    def is_done(self) -> bool:
        """Stop condition."""

    def step(self, backend_output: Optional[Any]) -> Tuple[Evidence, Dict[str, Any]]:
        """
        Decide what to do next.
        Returns:
          - evidence: {var:0/1} to enforce (could be empty)
          - info: optional debug/decision info (chosen var, score, etc.)
        """

    def solution(self) -> Any:
        """Return final solution."""

