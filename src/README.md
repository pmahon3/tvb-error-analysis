# Code

Run every script from the repository root, for example:

```sh
python3 -I src/sweep/sweep_maps.py results/runs/sweep/done results 0.005 0.02 0.005
```

Each script's first lines say what it does and how to call it. `./src/runall.sh` rebuilds every figure, table and the
write-up from the statistics kept in `results/runs`, without running a simulation.

## Folders

| Folder | What it holds |
|---|---|
| `core/` | The simulation itself. `mprjax.py` is the model and integrator (a JAX version of The Virtual Brain's Montbrió–Pazó–Roxin kernel, batched over parameter settings and seeds). `mprjax_run.py` runs one batch and computes the statistics; `mprjax_worker.py` is the job queue. `stats.py` holds the paper's summary statistics. `mpr_net.py` is an earlier NumPy version of the model, and `repo_paths.py` says where inputs and outputs are. |
| `sweep/` | The parameter sweep: job files, analysis, maps, burn-in checks, the region-by-region analysis (`sweep_per_region.py`), the test of whether nudged runs sit on the base curve (`sweep_matching.py`), and the write-up. |
| `validation/` | Checks that the JAX code reproduces TVB, step by step and in distribution, plus timing; the record of what happens before a run goes non-finite (`blowup_record.py`); a three-region check without noise (`chain_response_check.py`). |
| `tvb_reference/` | Reference runs in The Virtual Brain through the authors' group's companion code, at four integration steps; and what TVB's kernel does when all delays are zero. |
| `earlier_studies/` | Work that came before the sweep and is superseded where it disagrees: a first reimplementation, a two-state reduction, checks on Pillai & Jirsa (2017), and first attempts at the G-versus-η question (which found that the regime under study depended on the integration step). |

## Where the sweep's simulation code is

| Step | File |
|---|---|
| Write the jobs | `sweep/sweep_make_jobs.py` |
| Hand jobs to a GPU | `core/mprjax_worker.py` |
| Run one batch of about 1,000 simulations | `core/mprjax_run.py` |
| The model and integrator | `core/mprjax.py` |

In `core/mprjax.py`: `f` is the node equations, `gather` the delayed coupling between regions, `sample` the stochastic
Heun step. `build` adds the starting state, the noise, the BOLD model and the per-volume sums.

## How the simulations were run

Everything needed to repeat a run is in this repository except the two libraries and the container image, which are
public and pinned below.

**Reference runs in The Virtual Brain** (`tvb_reference/`)
- Software: `tvb-library` 2.10.0 (pip), on a Windows machine, CPU only.
- Our scripts call the companion code of the authors' group unchanged (`external/pci_fluidity/utils/simulation.py`),
  which calls TVB's numba Montbrió–Pazó–Roxin backend.
- The two kernel files TVB actually ran are kept for reading in `external/tvb_master/`; their checksums equal those of
  the installed files.
- Launched with `python tvb_ref_batch.py 6` (six runs at a time), then the two `tvb_ref_batch_seeds_*.py` scripts.

**JAX runs** (`core/`, `sweep/`, `validation/`)
- Software: container image `ghcr.io/ins-amu/vbjax@sha256:42d2e1aca2d590d7474688458b5d7623b93fadf11d7d82fa890179ff5f180ab9`
  (Python 3.10, jax 0.4.20, numpy 1.26.1), used only as a ready-made GPU environment. The vbjax library itself is not
  called by these runs; `core/mprjax.py` imports only JAX and NumPy.
- Hardware: one RTX 3060 and one RTX 3050, one worker per card. From the workers' logs, the RTX 3060 ran sweep
  batches `a_dt0.01_00`, `02`, `04`, `06`, `07`, `b_dt0.005_01` and `c_f64row_00`; the RTX 3050 ran `a_dt0.01_01`, `03`,
  `05`, `08` and `b_dt0.005_00`.
- Job files: `results/runs/sweep/jobs/*.json`. Each result file in `results/runs/sweep/done/` also carries the job that
  produced it.
- Launched with one container per card, each running the queue worker:
  `docker run -d --gpus all -e CUDA_VISIBLE_DEVICES=0 -v <folder>:/work -w /work <image> python3 mprjax_worker.py gpu0 q_sweep`
  (and the same with `1` and `gpu1`).

**Exactly which files ran.** The runs were made from a flat copy of these scripts. Commit `c1c91f3` holds them as run:
`mprjax_run.py`, `mprjax_worker.py`, `stats.py`, `tvb_ref_run.py`, `tvb_ref_batch.py` and `mpr_net.py` are
byte-identical there (checksums compared with the run machine). `mprjax.py` there differs only by an optional debug
output added after the sweep, off by default. Since then the files were moved into folders and reformatted; a fixed
small job gives identical statistics before and after.

**Earlier studies** ran on a laptop CPU with NumPy, or JAX with `vbjax` 0.0.19 for the node equations
(`earlier_studies/ridge_checks/ridge_v2_*.py`).

## Renamed scripts

`doc/LOG.md` refers to scripts by the names they had when the entries were written.

| Name in the log | Now |
|---|---|
| `sweep_post.py`, `sweep_post2.py`, `sweep_post3.py` | `sweep/sweep_describe.py`, `sweep_controls.py`, `sweep_confounding.py` |
| `burnin_check2.py` | `sweep/burnin_check_grid.py` |
| `tvb_ref_batch_more.py`, `tvb_ref_batch_more2.py` | `tvb_reference/tvb_ref_batch_seeds_9_32.py`, `tvb_ref_batch_seeds_33_96.py` |
| `mprjax_gather_bench2.py` | `validation/mprjax_history_copy_bench.py` |
| `ridge_v2_sde_check2.py` | `earlier_studies/ridge_checks/ridge_v2_sde_check_end_state.py` |

All other scripts kept their names.
