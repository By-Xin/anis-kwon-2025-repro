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
from sdp_relax_diag.data import load_market_data, rebalance_schedule, training_window, validation_segment
from sdp_relax_diag.ste import train_integer_forward_relaxed_backward
from sdp_relax_diag.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Train with exact integer forward and relaxed backward STE.")
    ap.add_argument("--config", default=str(ROOT / "configs/sdp_diagnostics/paper50.yaml"))
    ap.add_argument("--methods", nargs="+", default=["bigm", "socp", "sdp"])
    ap.add_argument("--cardinalities", nargs="+", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=1)
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
    result_path = out / "ste_integer_forward.csv"
    for widx, (reb, nxt) in enumerate(tqdm(windows, desc="windows")):
        f_train, r_train = training_window(md, reb, cfg)
        val = validation_segment(md, reb, nxt)
        for k in ks:
            for method in args.methods:
                try:
                    sub = train_integer_forward_relaxed_backward(method.lower(), f_train, r_train, val, cfg, int(k), rng, int(args.n_samples), int(args.epochs))
                    for row in sub:
                        row.update({"window": widx, "rebalance": reb.date(), "next_rebalance": nxt.date(), "cardinality": int(k), "n_samples": int(args.n_samples), "epochs_requested": int(args.epochs)})
                    rows.extend(sub)
                except Exception as exc:
                    rows.append({"window": widx, "rebalance": reb.date(), "cardinality": int(k), "method": method.lower(), "error": repr(exc)})
                    print(f"[WARN] STE failed method={method} window={widx} k={k}: {exc}")
                pd.DataFrame(rows).to_csv(result_path, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(result_path, index=False)
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    main()
