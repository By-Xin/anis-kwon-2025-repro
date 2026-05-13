#!/usr/bin/env python
"""Optional helper to download Fama-French five factors.

The paper used Kenneth French's Data Library via pandas_datareader. This script is
included for convenience; the official reproduction input remains data/factors_daily.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/factors_daily.csv")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2021-12-31")
    args = p.parse_args()
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        raise SystemExit("Install pandas-datareader first: pip install pandas-datareader") from e
    raw = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=args.start, end=args.end)[0]
    raw = raw.rename_axis("date").reset_index()
    cols = ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    raw[cols].to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
