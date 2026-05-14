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
from sdp_relax_diag.diagnostics import compare_to_exact
from sdp_relax_diag.factor import fit_factor_model
from sdp_relax_diag.metrics import realized_metrics
from sdp_relax_diag.portfolio import solve_cardinality
from sdp_relax_diag.relaxations import solve_relaxation
from sdp_relax_diag.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run relaxation quality and rounding diagnostics.")
    ap.add_argument("--config", default=str(ROOT / "configs/sdp_diagnostics/paper50.yaml"))
    ap.add_argument("--methods", nargs="+", default=["exact", "bigm", "socp", "sdp"])
    ap.add_argument("--cardinalities", nargs="+", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    rng = set_seed(int(cfg["project"].get("seed", 12345)))
    out = output_dir(cfg)
    save_config(cfg, out / "config_resolved.yaml")
    md = load_market_data(cfg)
    windows = rebalance_schedule(md, cfg, args.max_windows)
    ks = args.cardinalities or list(cfg["experiment"]["cardinalities"])
    methods = [m.lower() for m in args.methods]

    rows = []
    for widx, (reb, nxt) in enumerate(tqdm(windows, desc="windows")):
        f_train, r_train = training_window(md, reb, cfg)
        val = validation_segment(md, reb, nxt)
        fit = fit_factor_model(f_train, r_train, float(cfg["solver"].get("psd_jitter", 1e-8)))
        for k in ks:
            exact = solve_cardinality(fit.sigma, int(k), cfg["solver"])
            base = {
                "window": widx,
                "rebalance": reb.date(),
                "next_rebalance": nxt.date(),
                "cardinality": int(k),
                "exact_obj": exact.objective,
                "exact_status": exact.status,
                "exact_exact": exact.exact,
                "exact_support_size": len(exact.support),
            }
            base.update(realized_metrics(r_train, exact.weights, prefix="exact_train_"))
            if not val.empty:
                base.update(realized_metrics(val, exact.weights, prefix="exact_val_"))
            if "exact" in methods:
                rows.append({**base, "method": "exact"})
            for method in methods:
                if method == "exact":
                    continue
                try:
                    rel = solve_relaxation(method, fit, int(k), cfg["solver"])
                    diag = compare_to_exact(fit.sigma, int(k), exact.weights, exact.objective, rel.weights, rel.objective, rel.z, rel.W)
                    row = {**base, "method": method, "relax_status": rel.status, "relax_solve_time": rel.solve_time, **diag}
                    row.update(realized_metrics(r_train, rel.weights, prefix="relax_train_"))
                    if not val.empty:
                        row.update(realized_metrics(val, rel.weights, prefix="relax_val_"))
                    rows.append(row)
                except Exception as exc:
                    rows.append({**base, "method": method, "error": repr(exc)})
                    print(f"[WARN] {method} failed window={widx} k={k}: {exc}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "relaxation_quality.csv", index=False)
    print(f"Wrote {out/'relaxation_quality.csv'}")


if __name__ == "__main__":
    main()
