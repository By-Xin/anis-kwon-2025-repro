#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from sdp_relax_diag.config import apply_overrides, load_config
from sdp_relax_diag.data import load_market_data, validation_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/sdp_diagnostics/paper50.yaml"))
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    md = load_market_data(cfg)
    rep = validation_report(md, cfg)
    for k, v in rep.items():
        print(f"{k}: {v}")
    if not rep["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
