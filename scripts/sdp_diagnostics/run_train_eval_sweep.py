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
from sdp_relax_diag.train import train_one_window
from sdp_relax_diag.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate E2E layers across sample-size and epoch grids.")
    ap.add_argument("--config", default=str(ROOT / "configs/sdp_diagnostics/paper50.yaml"))
    ap.add_argument("--methods", nargs="+", default=["bigm", "socp", "sdp"])
    ap.add_argument("--cardinalities", nargs="+", type=int, default=None)
    ap.add_argument("--sample-grid", nargs="+", type=int, default=[128, 512, 2000])
    ap.add_argument("--epoch-grid", nargs="+", type=int, default=[1, 4, 16])
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

    for widx, (reb, nxt) in enumerate(tqdm(windows, desc="windows")):
        f_train, r_train = training_window(md, reb, cfg)
        val = validation_segment(md, reb, nxt)
        for k in ks:
            for method in args.methods:
                for n_samples in args.sample_grid:
                    for epochs in args.epoch_grid:
                        try:
                            res = train_one_window(method.lower(), f_train, r_train, val, cfg, int(k), rng, int(n_samples), int(epochs))
                            hist = res.pop("history")
                            row = {"window": widx, "rebalance": reb.date(), "next_rebalance": nxt.date(), "cardinality": int(k), "method": method.lower(), "n_samples": int(n_samples), "epochs": int(epochs), **res}
                            rows.append(row)
                            # Save epoch history as denormalized rows too.
                            for h in hist:
                                hr = {"window": widx, "rebalance": reb.date(), "cardinality": int(k), "method": method.lower(), "n_samples": int(n_samples), "epochs_requested": int(epochs), **h}
                                rows.append({**hr, "is_epoch_history": True})
                        except Exception as exc:
                            rows.append({"window": widx, "rebalance": reb.date(), "cardinality": int(k), "method": method.lower(), "n_samples": int(n_samples), "epochs": int(epochs), "error": repr(exc)})
                            print(f"[WARN] train failed method={method} J={n_samples} epochs={epochs} window={widx} k={k}: {exc}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "train_eval_sweep.csv", index=False)
    print(f"Wrote {out/'train_eval_sweep.csv'}")


if __name__ == "__main__":
    main()
