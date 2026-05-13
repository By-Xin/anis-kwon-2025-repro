# Results Summary: 2015-2020

Generated from the current WSL reproduction run. Data files are committed in `data/`; full local result artifacts are under `results/` and remain ignored by git.

## Environment

- Python 3.10.20 in WSL `.venv`
- `cvxpy==1.4.4`
- `cvxpylayers==0.1.6`
- `torch==2.3.1+cu121`
- `gurobipy==13.0.2`
- `scs==3.2.2`
- Gurobi WLS academic license was available and all test-time MIQP solves below returned `optimal`.

Note: the paper reports SCS 3.2.1. That exact version was unavailable from the configured pip index, so these runs use SCS 3.2.2.

## Completed Runs

| Run | Cardinalities | Rebalance rows | Test MIQP status | CVXPYLayer solve failures | Runtime seconds |
|---|---:|---:|---|---:|---:|
| `nominal`, `linreg` | 10, 15, 20 | 144 | 144 optimal | n/a | 8.24 |
| `e2e_m` | 10 | 24 | 24 optimal | 0 | 2422.42 |
| `e2e_m` | 15 | 24 | 24 optimal | 0 | 2211.37 |
| `e2e_m` | 20 | 24 | 24 optimal | 0 | 2069.05 |
| `e2e_socp` | 10 | 24 | 24 optimal | 0 | 5583.38 |
| `e2e_socp` | 15 | 24 | 24 optimal | 0 | 5286.71 |
| `e2e_socp` | 20 | 24 | 24 optimal | 0 | 5043.09 |

`e2e_sdp` was not run overnight; it should be treated as a separate compute job.

## Metrics

| method | k | avgRet | annRet | annVol | Sharpe | maxDD | Calmar | Sortino | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nominal | 10 | 0.000333 | 0.129235 | 0.170102 | 0.759750 | 0.376302 | 0.343434 | 1.436555 | 7.466131 |
| nominal | 15 | 0.000335 | 0.130246 | 0.168091 | 0.774852 | 0.373979 | 0.348269 | 1.477928 | 6.089682 |
| nominal | 20 | 0.000334 | 0.129597 | 0.168507 | 0.769088 | 0.376823 | 0.343920 | 1.459954 | 6.100468 |
| linreg | 10 | 0.000294 | 0.113089 | 0.173150 | 0.653126 | 0.399794 | 0.282868 | 1.155979 | 5.056240 |
| linreg | 15 | 0.000306 | 0.117944 | 0.170041 | 0.693619 | 0.384399 | 0.306827 | 1.271031 | 4.894878 |
| linreg | 20 | 0.000312 | 0.120720 | 0.169230 | 0.713347 | 0.380110 | 0.317592 | 1.322522 | 4.580014 |
| e2e_m | 10 | 0.000470 | 0.187216 | 0.239115 | 0.782953 | 0.330817 | 0.565920 | 1.113984 | 14.388514 |
| e2e_m | 15 | 0.000492 | 0.196583 | 0.241250 | 0.814853 | 0.331119 | 0.593693 | 1.152984 | 13.946172 |
| e2e_m | 20 | 0.000501 | 0.200707 | 0.244398 | 0.821231 | 0.347292 | 0.577921 | 1.136285 | 13.794504 |
| e2e_socp | 10 | 0.000516 | 0.207090 | 0.263808 | 0.785003 | 0.352581 | 0.587355 | 1.012076 | 15.538145 |
| e2e_socp | 15 | 0.000475 | 0.189418 | 0.264078 | 0.717282 | 0.347061 | 0.545778 | 0.911004 | 16.417955 |
| e2e_socp | 20 | 0.000464 | 0.184423 | 0.265213 | 0.695378 | 0.348671 | 0.528932 | 0.869207 | 15.574135 |

## Local Result Directories

- `results/reproduce_2015_2020_baseline/`
- `results/reproduce_2015_2020_e2e_m_k10/`
- `results/reproduce_2015_2020_e2e_m_k15/`
- `results/reproduce_2015_2020_e2e_m_k20/`
- `results/reproduce_2015_2020_e2e_socp_k10/`
- `results/reproduce_2015_2020_e2e_socp_k15/`
- `results/reproduce_2015_2020_e2e_socp_k20/`

