"""Run recipes: which heads a run trains.

    configs/marginals.yaml   four scalar heads and a (SFR, M*) joint
    configs/rates.yaml       the (lambda_P2, lambda_P3) joint
    configs/joint4.yaml      the (lambda_P2, lambda_P3, SFR, M*) joint

The reported runs "share this architecture and optimizer and differ only in
their heads", so a recipe carries a name and a head list and nothing else. The
optimizer is `TRAINING` below, transcribed from the paper; the architecture
constants live beside the code they describe, in `encoder.py`, `flows.py` and
`poisson.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .data import TARGETS


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Training:
    """AdamW and the schedule, one setting for every run."""

    betas: tuple[float, float] = (0.95, 0.999)
    lr_readout: float = 3e-4        # the readout MLPs and the CLS token
    lr_flow: float = 1e-3
    lr_adapter: float = 3e-5
    wd_readout: float = 1e-4        # the readout MLPs and the flows
    wd_adapter: float = 0.1
    wd_cls: float = 0.0
    batch_size: int = 896
    grad_clip: float = 5.0
    max_epochs: int = 40
    patience: int = 5
    seed: int = 42


TRAINING = Training()


@dataclass(frozen=True)
class Head:
    """One readout MLP and one flow over `targets`; several targets make a joint."""

    name: str
    targets: tuple[str, ...]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(TARGETS[t].kind for t in self.targets)

    @property
    def is_joint(self) -> bool:
        return len(self.targets) > 1


@dataclass(frozen=True)
class Run:
    name: str
    heads: tuple[Head, ...]

    @property
    def targets(self) -> tuple[str, ...]:
        """Every target any head trains on, in registry order."""
        used = {t for head in self.heads for t in head.targets}
        return tuple(name for name in TARGETS if name in used)

    def head(self, name: str) -> Head:
        for head in self.heads:
            if head.name == name:
                return head
        raise ConfigError(f"no head {name!r} in run {self.name!r}")


def parse_run(spec: dict) -> Run:
    """Validate a recipe mapping: `name`, and `heads` from head name to target list."""
    unknown = set(spec) - {"name", "heads"}
    if unknown:
        raise ConfigError(f"unknown recipe keys {sorted(unknown)}")
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError("a recipe needs a non-empty `name`")
    heads_spec = spec.get("heads")
    if not isinstance(heads_spec, dict) or not heads_spec:
        raise ConfigError("a recipe needs a non-empty `heads` mapping")
    heads = []
    for head_name, targets in heads_spec.items():
        if isinstance(targets, str) or not isinstance(targets, (list, tuple)) or not targets:
            raise ConfigError(f"head {head_name!r}: targets must be a non-empty list")
        bad = [t for t in targets if t not in TARGETS]
        if bad:
            raise ConfigError(f"head {head_name!r}: unknown targets {bad}; "
                              f"known are {sorted(TARGETS)}")
        if len(set(targets)) != len(targets):
            raise ConfigError(f"head {head_name!r}: repeated target")
        heads.append(Head(str(head_name), tuple(targets)))
    return Run(name, tuple(heads))


def load_run(path: str | Path) -> Run:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no run recipe at {path}")
    return parse_run(yaml.safe_load(path.read_text()))
