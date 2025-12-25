from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List
import warnings

Config = dict[str, Any]

def _get(cfg: Config, path: str, default=None):
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _set(cfg: Config, path: str, value: Any) -> Tuple[bool, Any]:
    """Set cfg[path]=value. Returns (changed, old_value_or_None)."""
    keys = path.split(".")
    cur: Any = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    leaf = keys[-1]
    old = cur.get(leaf, None)
    if old != value:
        cur[leaf] = value
        return True, old
    return False, old

@dataclass(frozen=True)
class Rule:
    """Boolean predicate over a config."""
    pred: Callable[[Config], bool]
    desc: str = ""

    def __and__(self, other: "Rule") -> "Rule":
        return Rule(
            pred=lambda cfg: self.pred(cfg) and other.pred(cfg),
            desc=_join_desc(self, other, op="AND"),
        )

    def __or__(self, other: "Rule") -> "Rule":
        return Rule(
            pred=lambda cfg: self.pred(cfg) or other.pred(cfg),
            desc=_join_desc(self, other, op="OR"),
        )

    def __invert__(self) -> "Rule":
        return Rule(
            pred=lambda cfg: not self.pred(cfg),
            desc=f"NOT({self.desc})" if self.desc else "NOT(<rule>)",
        )

    def __call__(self, cfg: Config) -> bool:
        return self.pred(cfg)

def _join_desc(a: Rule, b: Rule, op: str) -> str:
    if a.desc and b.desc:
        return f"({a.desc} {op} {b.desc})"
    if a.desc:
        return f"({a.desc} {op} <rule>)"
    if b.desc:
        return f"(<rule> {op} {b.desc})"
    return f"(<rule> {op} <rule>)"

def Eq(path: str, value: Any) -> Rule:
    return Rule(
        pred=lambda cfg: _get(cfg, path) == value,
        desc=f"{path} == {value!r}",
    )

def In_(path: str, values: set[Any] | list[Any] | tuple[Any, ...]) -> Rule:
    vals = set(values)
    return Rule(
        pred=lambda cfg: _get(cfg, path) in vals,
        desc=f"{path} in {sorted(vals)!r}",
    )

def Exists(path: str) -> Rule:
    return Rule(
        pred=lambda cfg: _get(cfg, path, default=_MISSING) is not _MISSING,
        desc=f"exists({path})",
    )

class _Missing: pass
_MISSING = _Missing()

import warnings
from dataclasses import dataclass
from typing import Any, Callable, List

@dataclass(frozen=True)
class ConfigRule:
    name: str
    applies: Callable[[Config], bool]
    patch: Callable[[Config], List[str]]

def _rule_force(values: dict[str, Any], when: Rule, name: str | None = None) -> ConfigRule:
    def patch(cfg: Config) -> List[str]:
        out: List[str] = []
        for (k, v) in values.items():
            changed, old = _set(cfg, k, v)
            if changed:
                out.append(f"{k}: {old} -> {v} (required by {when.desc})")
        return out

    return ConfigRule(
        name=name or f"force: {when.desc} -> {values}",
        applies=lambda cfg: when(cfg),
        patch=patch,
    )

def apply_config_policies(cfg: Config) -> None:
    changes: List[str] = []
    for rule in MODEL_RULES:
        if rule.applies(cfg):
            changes.extend(rule.patch(cfg))

    if changes:
        msg = "Config was adjusted to satisfy model requirements:\n  - " + "\n  - ".join(changes)
        warnings.warn(msg, stacklevel=2)

MODEL_RULES: list[ConfigRule] = [
    # Force hetero for hetero models
    _rule_force(
        {"data.hetero": True},
        when=In_("model.name", {"HeteroGCNVarLogits", "Hetero_GCN_MRF_v4"}),
        name="hetero-models-force-data.hetero",
    ),
    _rule_force(
        {"data.hetero": False},
        when=In_("model.name", {"GCNVarLogits", "GCN_MRF_v4"}),
        name="hetero-models-force-data.hetero",
    ),

    #_rule_force(
    #    {"data.precompute.lbp": True},
    #    when=In_("model.name", ["GCN_MRF_v4", "Hetero_GCN_MRF_v4"]) & Eq("model.args.inference", "loopy_belief_propagation"),
    #    name="lbp-inference-requires-precompute.lbp",
    #),
    #_rule_force(
    #    {"data.precompute.nmf": True},
    #    when=In_("model.name", ["GCN_MRF_v4", "Hetero_GCN_MRF_v4"]) & Eq("model.args.inference", "naive_mean_field"),
    #    name="nmf-inference-requires-precompute.nmf",
    #),

]
