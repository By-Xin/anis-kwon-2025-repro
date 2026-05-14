#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from sdp_relax_diag.config import apply_overrides, load_config, output_dir, save_config
from sdp_relax_diag.data import load_market_data, rebalance_schedule, training_window
from sdp_relax_diag.gradient import gradient_alignment_for_window
from sdp_relax_diag.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare relaxation autograd gradients with finite-difference exact-integer loss directions.")
    ap.add_argument("--config", default=str(ROOT / "configs/sdp_diagnostics/paper50.yaml"))
    ap.add_argument("--methods", nargs="+", default=["bigm", "socp", "sdp"])
    ap.add_argument("--cardinalities", nargs="+", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=1)
    ap.add_argument("--n-bootstrap", type=int, default=16)
    ap.add_argument("--n-directions", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    out = output_dir(cfg)
    save_config(cfg, out / "config_resolved.yaml")
    rng = set_seed(int(cfg["project"].get("seed", 12345)))
    md = load_market_data(cfg)
    windows = rebalance_schedule(md, cfg, args.max_windows)
    ks = args.cardinalities or list(cfg["experiment"]["cardinalities"])
    rows = []

    for widx, (reb, nxt) in enumerate(tqdm(windows, desc="windows")):
        f_train, r_train = training_window(md, reb, cfg)
        for k in ks:
            for method in args.methods:
                try:
                    sub = gradient_alignment_for_window(method.lower(), f_train, r_train, cfg, int(k), rng, int(args.n_bootstrap), int(args.n_directions), float(args.eps))
                    for row in sub:
                        row.update({"window": widx, "rebalance": reb.date(), "next_rebalance": nxt.date(), "n_bootstrap": args.n_bootstrap, "n_directions": args.n_directions, "eps": args.eps})
                    rows.extend(sub)
                except Exception as exc:
                    rows.append({"window": widx, "rebalance": reb.date(), "cardinality": int(k), "method": method.lower(), "error": repr(exc)})
                    print(f"[WARN] gradient alignment failed method={method} window={widx} k={k}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "gradient_alignment.csv", index=False)
    print(f"Wrote {out/'gradient_alignment.csv'}")


if __name__ == "__main__":
    main()
