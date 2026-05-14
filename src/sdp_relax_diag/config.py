from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Empty config: {path}")
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def parse_scalar(s: str) -> Any:
    low = s.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if any(c in low for c in [".", "e"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, raw = item.split("=", 1)
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = parse_scalar(raw)
    return out


def output_dir(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["project"]["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p
