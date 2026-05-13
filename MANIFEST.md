# Bundle manifest

- `README.md` — run instructions.
- `docs/DATA_REQUEST.md` — data request/specification sheet for the user.
- `docs/COMPUTE_REQUEST.md` — compute and software requirements.
- `docs/REPRODUCTION_NOTES.md` — experimental design alignment and caveats.
- `configs/reproduce_2015_2020.yaml` — default reproduction config matching 24 quarterly periods.
- `configs/reproduce_2015_2021.yaml` — alternate wider OOS span.
- `configs/smoke.yaml` — synthetic smoke-test config.
- `scripts/check_data.py` — validate user-provided data.
- `scripts/run_reproduction.py` — run rolling backtest / training.
- `scripts/run_smoke_test.py` — generate synthetic data and run a lightweight baseline check.
- `scripts/download_ff_factors.py` — optional Fama-French factor downloader.
- `scripts/slurm_run_array.sh` — example cluster array job.
- `src/e2e_cardinality_portfolio/` — core implementation.
