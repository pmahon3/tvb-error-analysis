# Code

Run every script from the repository root, for example:

```sh
python3 -I src/7_parameter_sweep/sweep_maps.py results/runs/sweep/done results 0.005 0.02 0.005
```

Each script's first lines say what it does and how to call it. `./src/runall.sh` rebuilds every figure, table and the
write-up from the statistics kept in `results/runs`, without running a simulation.

## Folders

| Folder | What it holds |
|---|---|
| `common/` | Modules shared by several studies: `mprjax.py` (JAX version of The Virtual Brain's Montbrió–Pazó–Roxin kernel, batched over parameter settings and seeds), `mpr_net.py` (earlier NumPy version of the model, with a Balloon–Windkessel BOLD model), `stats.py` (the paper's summary statistics, written from its Methods), `repo_paths.py` (where inputs and outputs are) |
| `1_first_reimplementation/` | First simulations of the model over (G, σ), a surrogate test of the selection statistics, and the two essay figures |
| `2_two_state_reduction/` | A two-state (up/down) reduction of the network and the escape rates it needs |
| `3_pillai_jirsa/` | Lesion and readout checks on Pillai & Jirsa (2017) |
| `4_ridge_checks/` | Do the statistics separate G from η? First attempts. They found that the regime under study depended on the integration step |
| `5_tvb_reference/` | Reference runs in The Virtual Brain through the authors' group's companion code, at four integration steps; and what TVB's kernel does when all delays are zero |
| `6_jax_port/` | The batched GPU runner built on `common/mprjax.py`, its job queue, timing, and the checks against TVB (step by step, and in distribution) |
| `7_parameter_sweep/` | The sweep of 2026-11-24: job files, analysis, maps, burn-in checks, and the write-up |

The folders are numbered in the order the work was done. Later studies supersede earlier ones where they disagree;
`doc/LOG.md` has the history.

## How the simulations were run

Everything needed to repeat a run is in this repository except the two libraries and the container image, which are
public and pinned below.

**Reference runs in The Virtual Brain** (`5_tvb_reference/`)
- Software: `tvb-library` 2.10.0 (pip), on a Windows machine, CPU only.
- Our scripts call the companion code of the authors' group unchanged (`external/pci_fluidity/utils/simulation.py`),
  which calls TVB's numba Montbrió–Pazó–Roxin backend.
- The two kernel files TVB actually ran are kept for reading in `external/tvb_master/`; their checksums equal those of
  the installed files.
- Launched with `python tvb_ref_batch.py 6` (six runs at a time), then the two `tvb_ref_batch_seeds_*.py` scripts.

**JAX runs** (`6_jax_port/`, `7_parameter_sweep/`)
- Software: container image `ghcr.io/ins-amu/vbjax@sha256:42d2e1aca2d590d7474688458b5d7623b93fadf11d7d82fa890179ff5f180ab9`
  (Python 3.10, jax 0.4.20, numpy 1.26.1), used only as a ready-made GPU environment. The vbjax library itself is not
  called by these runs; the kernel is `common/mprjax.py`, which imports only JAX and NumPy.
- Hardware: one RTX 3060 and one RTX 3050, one worker per card.
- Job files: `results/runs/sweep/jobs/*.json`, written by `7_parameter_sweep/sweep_make_jobs.py`. Each result file in
  `results/runs/sweep/done/` also carries the job that produced it.
- Launched with one container per card, each running the queue worker:
  `docker run -d --gpus all -e CUDA_VISIBLE_DEVICES=0 -v <folder>:/work -w /work <image> python3 mprjax_worker.py gpu0 q_sweep`
  (and the same with `1` and `gpu1`).

**Exactly which files ran.** The runs were made from a flat copy of these scripts. Commit `c1c91f3` holds them as run:
`mprjax_run.py`, `mprjax_worker.py`, `stats.py`, `tvb_ref_run.py`, `tvb_ref_batch.py` and `mpr_net.py` are
byte-identical there (checksums compared with the run machine). `mprjax.py` there differs only by an optional debug
output added after the sweep, off by default. Since then the files were moved into folders and reformatted; a fixed
small job gives identical statistics before and after.

**Earlier studies** (`1_`–`4_`) ran on a laptop CPU with NumPy, or JAX with `vbjax` 0.0.19 for the node equations
(`4_ridge_checks/ridge_v2_*.py`).

## Renamed scripts

`doc/LOG.md` refers to scripts by the names they had when the entries were written.

| Name in the log | Now |
|---|---|
| `sweep_post.py`, `sweep_post2.py`, `sweep_post3.py` | `7_parameter_sweep/sweep_describe.py`, `sweep_controls.py`, `sweep_confounding.py` |
| `burnin_check2.py` | `7_parameter_sweep/burnin_check_grid.py` |
| `tvb_ref_batch_more.py`, `tvb_ref_batch_more2.py` | `5_tvb_reference/tvb_ref_batch_seeds_9_32.py`, `tvb_ref_batch_seeds_33_96.py` |
| `mprjax_gather_bench2.py` | `6_jax_port/mprjax_history_copy_bench.py` |
| `ridge_v2_sde_check2.py` | `4_ridge_checks/ridge_v2_sde_check_end_state.py` |

All other scripts kept their names and moved into the folder of their study.
