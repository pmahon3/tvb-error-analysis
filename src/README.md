# Code

All scripts sit in this one folder and are run from here. `repo_paths.py` finds the inputs (`../data`, `../external`)
and the per-run output (`../results/runs`). Each script's first lines say what it does and how to call it.

`./runall.sh` rebuilds every figure, table and the write-up from the statistics kept in `../results/runs`. It runs no
simulation.

## Shared modules

| File | What it is |
|---|---|
| `mprjax.py` | JAX version of The Virtual Brain's Montbrió–Pazó–Roxin kernel, batched over parameter settings and seeds |
| `mpr_net.py` | Earlier NumPy reimplementation of the model, with a Balloon–Windkessel BOLD model |
| `stats.py` | The paper's summary statistics, written from its Methods |
| `repo_paths.py` | Locates input and output folders |

## By study, in the order the work was done

| Scripts | Study |
|---|---|
| `run_sweep.py`, `rebold.py`, `g1_surrogate.py`, `fig_g1.py`, `fig_f2.py` | First reimplementation: (G, σ) batches, surrogate test of the selection statistics, essay figures |
| `ising_net.py`, `fit_rates.py`, `escape_times.py` | Two-state reduction of the network |
| `pillai_jirsa_*.py` | Lesion and readout checks on Pillai & Jirsa (2017) |
| `ridge_check.py`, `ridge_diag.py`, `ridge_v2_*.py` | Do the statistics separate G from η? First attempts, including the finding that the regime studied depended on the integration step |
| `tvb_ref_*.py`, `tvb_init_check.py` | Reference runs in The Virtual Brain through the authors' group's companion code, at four integration steps |
| `tvb_zero_delay_*.py` | What TVB's numba kernel does when all delays are zero |
| `mprjax_match.py`, `mprjax_m2_score.py`, `stats_path_check.py` | Checks of the JAX version against TVB: step by step, and in distribution |
| `mprjax_run.py`, `mprjax_worker.py`, `mprjax_bench.py`, `mprjax_gather_bench*.py`, `mprjax_v1_perstep.py` | Batched GPU runner, job queue, timing |
| `sweep_make_jobs.py`, `sweep_pilot_score.py`, `sweep_maps.py`, `sweep_post*.py`, `sweep_nonfinite_check.py` | The parameter sweep of 2026-11-24 and its analysis |
| `burnin_check*.py`, `dump_traces.py` | How long runs take to settle after the discarded start |
| `writeup_sweep.py` | Builds the figures in `../results/figures/report` and `../doc/sweep_writeup.pdf` |

Scripts whose usage line says "tower" were run on a separate GPU machine from a flat copy of this folder; the folder
names in them (`runs_A`, `runs_port`, `q_sweep`) are that machine's.
