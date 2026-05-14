#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def summarize_relaxation(rd: Path) -> None:
    path = rd / "relaxation_quality.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "error" in df.columns:
        good = df[df["error"].isna()] if df["error"].notna().any() else df
    else:
        good = df
    metrics = [c for c in ["gap_to_exact", "bound_violation", "relax_portfolio_gap_to_exact", "rounded_gap_to_exact", "topk_overlap", "rounded_support_overlap", "l1_to_exact", "frac_sum", "rank_ratio1", "W_minus_ww_norm", "relax_solve_time"] if c in good.columns]
    if not metrics:
        return
    summary = good[good["method"] != "exact"].groupby(["cardinality", "method"])[metrics].agg(["mean", "std", "median", "count"])
    summary.to_csv(rd / "summary_relaxation_quality.csv")
    print("=== relaxation quality summary ===")
    print(summary)
    if "gap_to_exact" in good.columns:
        fig = plt.figure()
        good[good["method"] != "exact"].boxplot(column="gap_to_exact", by="method")
        plt.title("Relaxation gap to exact")
        plt.suptitle("")
        plt.ylabel("(exact - relaxation) / exact")
        fig.savefig(rd / "fig_relaxation_gap.png", bbox_inches="tight", dpi=180)
        plt.close(fig)
    if "topk_overlap" in good.columns:
        fig = plt.figure()
        good[good["method"] != "exact"].boxplot(column="topk_overlap", by="method")
        plt.title("Top-k overlap with exact support")
        plt.suptitle("")
        plt.ylabel("overlap")
        fig.savefig(rd / "fig_topk_overlap.png", bbox_inches="tight", dpi=180)
        plt.close(fig)


def summarize_gradient(rd: Path) -> None:
    path = rd / "gradient_alignment.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    good = df[df.get("error", pd.Series([None] * len(df))).isna()] if "error" in df.columns else df
    cols = [c for c in ["direction_corr_within_window", "sign_agreement_rate_within_window", "grad_norm", "grad_nan_count", "bad_layer_solves"] if c in good.columns]
    if not cols:
        return
    summary = good.groupby(["cardinality", "method"])[cols].mean()
    summary.to_csv(rd / "summary_gradient_alignment.csv")
    print("=== gradient alignment summary ===")
    print(summary)
    if "direction_corr_within_window" in good.columns:
        fig = plt.figure()
        good.boxplot(column="direction_corr_within_window", by="method")
        plt.title("Correlation: relaxation gradient vs exact-integer FD")
        plt.suptitle("")
        plt.ylabel("correlation")
        fig.savefig(rd / "fig_gradient_corr.png", bbox_inches="tight", dpi=180)
        plt.close(fig)


def summarize_train(rd: Path) -> None:
    path = rd / "train_eval_sweep.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "is_epoch_history" in df.columns:
        final = df[df["is_epoch_history"].isna()]
    else:
        final = df
    if "error" in final.columns:
        final = final[final["error"].isna()]
    if final.empty:
        return
    cols = [c for c in ["final_exact_val_sharpe", "final_train_loss_mean", "final_beta_drift", "final_psi_drift"] if c in final.columns]
    if not cols:
        return
    summary = final.groupby(["method", "n_samples", "epochs"])[cols].mean()
    summary.to_csv(rd / "summary_train_eval_sweep.csv")
    print("=== train/eval sweep summary ===")
    print(summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.results_dir)
    summarize_relaxation(rd)
    summarize_gradient(rd)
    summarize_train(rd)


if __name__ == "__main__":
    main()
