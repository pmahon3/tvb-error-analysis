"""A short note on the grey cells of the sweep's maps: where they are, and what happens there.
(doc/LOG.md, 2026-11-24, 21:22, part B, and its outcomes.) Every number is computed here from the run files.

Usage (from the repository root):
    python3 -I src/sweep/writeup_grey_cells.py results/runs/sweep/done results/runs/tvb_reference \
        results/runs/validation results doc/grey_cells.pdf
Writes its figures to <results_dir>/figures/grey_cells/ and builds the PDF with pdflatex.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import glob
import json
import os
import site
import warnings
from collections import defaultdict

sys.path.append(
    site.getusersitepackages()
)  # the plotting library lives in the user site; -I hides it
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import sweep_maps
from report_common import C_F32, C_TVB, HOLE, INK, INK2, SEQ, SURF, build_pdf
from sweep_maps import D0, ETA0, J0, NAMES, STATS

sweep_folder, tvb_folder, validation_folder, results_folder, out_path = sys.argv[1:6]
figure_folder = os.path.join(results_folder, "figures", "grey_cells")
os.makedirs(figure_folder, exist_ok=True)
figures = {}


def save(fig, name):
    path = os.path.join(figure_folder, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures[name] = path


# ---------------------------------------------------------------- the sweep's runs
batches = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sweep_folder, "*.json")))]
all_runs = [run for batch in batches for run in batch["runs"]]
full_step = [r for r in all_runs if r["dt"] == 0.01 and not r.get("x64")]
half_step = [r for r in all_runs if r["dt"] == 0.005]
G_values = sorted({round(r["G"], 2) for r in full_step})
sigma_values = sorted({round(r["nsig"], 3) for r in full_step})
G_array, sigma_array = np.array(G_values), np.array(sigma_values)
shape = (len(sigma_values), len(G_values))

lost = np.zeros(shape, int)  # runs lost per cell, of 28 (7 settings x 4 seeds)
made = np.zeros(shape, int)
largest_rate = np.full(
    shape, np.nan
)  # largest output-sample rate reached by a finite run in the cell
for r in full_step:
    i, j = sigma_values.index(round(r["nsig"], 3)), G_values.index(round(r["G"], 2))
    made[i, j] += 1
    if r["finite"]:
        largest_rate[i, j] = np.nanmax([largest_rate[i, j], r["max_r"]])
    else:
        lost[i, j] += 1
n_lost, n_made = int(lost.sum()), int(made.sum())
half_lost = sum(not r["finite"] for r in half_step)

# ---------------------------------------------------------------- which cells are grey in the cosine maps, and why
cells, _ = sweep_maps.load(sweep_folder)
mean, se, sd, _ = sweep_maps.grid(cells, G_values, sigma_values)
G_effect, G_effect_se = sweep_maps.central(mean, se, G_array, 2)
scale = np.where(sd > 0, sd, np.nan)
length = np.sqrt(np.nansum((G_effect / scale) ** 2, 0))
length_se = np.sqrt(np.nansum((G_effect_se / scale) ** 2, 0))
up_occupancy = mean[STATS.index("m2_occ_up07")]
eta_up = sweep_maps.grid(cells, G_values, sigma_values, eta=ETA0 + 0.005, J=J0, Delta=D0)[0]
eta_down = sweep_maps.grid(cells, G_values, sigma_values, eta=ETA0 - 0.005, J=J0, Delta=D0)[0]
eta_effect = (eta_up - eta_down) / 0.01
estimable = np.isfinite(G_effect / scale).all(0) & np.isfinite(eta_effect / scale).all(0)
strong = length > 3 * length_se
mixed = (up_occupancy > 0.02) & (up_occupancy < 0.90)
used = estimable & strong & mixed

# a grey cell is put down to lost runs if a run was lost there or in a neighbouring cell along G (the G derivative
# uses the neighbours); otherwise it is quiet
near_loss = lost > 0
near_loss[:, 1:] |= lost[:, :-1] > 0
near_loss[:, :-1] |= lost[:, 1:] > 0
category = np.where(used, 0, np.where(near_loss, 2, 1))
n_used, n_quiet, n_holes = (int((category == k).sum()) for k in (0, 1, 2))

# the quiet corner in numbers
quiet = category == 1
quiet_corner = quiet & (
    G_array[None, :] <= 0.53 + 1e-9
)  # the low-G block; the rest are isolated cells
RATE = STATS.index("m1_mean_r")
example = (sigma_values.index(0.022), G_values.index(0.45))
example_change = G_effect[RATE][example] * 0.01
example_se = G_effect_se[RATE][example] * 0.01
working_point = (sigma_values.index(0.03), G_values.index(0.54))

# ---------------------------------------------------------------- figure 1: where
extent = [
    G_values[0] - 0.005,
    G_values[-1] + 0.005,
    sigma_values[0] - 0.002,
    sigma_values[-1] + 0.002,
]
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
ax = axes[0]
ax.imshow(
    category,
    origin="lower",
    extent=extent,
    aspect="auto",
    cmap=ListedColormap(["#f0efec", C_TVB, C_F32]),
    vmin=0,
    vmax=2,
)
ax.set_title("Why a cell is grey in the cosine maps")
for colour, label in (
    ("#f0efec", "used"),
    (C_TVB, "unresponsive"),
    (C_F32, "a run diverged here\nor in the adjacent cell"),
):
    ax.scatter([], [], marker="s", s=60, color=colour, edgecolor=INK2, linewidths=0.4, label=label)
ax.legend(
    loc="center right", fontsize=7, frameon=True, facecolor=SURF, edgecolor=INK2, framealpha=0.95
)
ax = axes[1]
im = ax.imshow(
    np.where(lost > 0, lost, np.nan), origin="lower", extent=extent, aspect="auto", cmap=SEQ, vmin=0
)
fig.colorbar(im, ax=ax, shrink=0.85)
ax.set_title("Runs that diverged per cell (of 28)")
ax = axes[2]
im = ax.imshow(largest_rate, origin="lower", extent=extent, aspect="auto", cmap=SEQ, vmin=0)
fig.colorbar(im, ax=ax, shrink=0.85)
ax.set_title("Largest rate reached by a finite run")
for ax in axes:
    ax.set_facecolor(SURF)
    ax.grid(False)
    ax.set_xlabel("G")
    ax.set_ylabel("σ")
    ax.plot([0.54], [0.030], marker="+", color=INK, ms=10, mew=1.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
fig.tight_layout()
save(fig, "g1_where.png")

# ---------------------------------------------------------------- when the lost runs failed, and what they share
per_run = np.load(os.path.join(os.path.dirname(sweep_folder.rstrip("/")), "node_stats_sweep.npz"))
failed = np.flatnonzero(~per_run["finite"])
failure_volume = per_run["first_bad_tr"][failed]
events = defaultdict(
    list
)  # runs that share a seed share their noise: (seed, step, volume) -> settings
for i in failed:
    key = (int(per_run["seed"][i]), float(per_run["dt"][i]), int(per_run["first_bad_tr"][i]))
    events[key].append((float(per_run["G"][i]), float(per_run["sigma"][i])))
shared_events = sorted((k, v) for k, v in events.items() if len(v) >= 3)
T_events = [
    [
        str(seed),
        str(volume),
        str(len(settings)),
        f"{min(g for g, _ in settings):.2f}–{max(g for g, _ in settings):.2f}",
        f"{min(s for _, s in settings):.3f}–{max(s for _, s in settings):.3f}",
    ]
    for (seed, _, volume), settings in shared_events
]

lost_by_sigma = {
    sigma: (int(lost[i].sum()), int(made[i].sum())) for i, sigma in enumerate(sigma_values)
}
T_sigma = [
    [f"{sigma:.3f}", f"{a} of {b}"] for sigma, (a, b) in lost_by_sigma.items() if sigma >= 0.034
]
first_sigma_with_loss = min(sigma for sigma, (a, _) in lost_by_sigma.items() if a > 0)
lowest_G_with_loss = min(G_values[j] for j in range(len(G_values)) if lost[:, j].sum() > 0)

# ---------------------------------------------------------------- repeat runs at one cell that lost runs
T_repeat, lost_seeds = [], {}
for file_name, label in (
    ("h1_f32_dt0.01.json", "JAX, single precision, step 0.01"),
    ("h2_f64_dt0.01.json", "JAX, double precision, step 0.01"),
    ("h3_f32_dt0.005.json", "JAX, single precision, step 0.005"),
):
    runs = json.load(open(os.path.join(validation_folder, "hole_cell", file_name)))["runs"]
    lost_seeds[file_name] = sorted(r["seed"] for r in runs if not r["finite"])
    T_repeat.append([label, f"{len(lost_seeds[file_name])} of {len(runs)}"])
jax_cell = json.load(open(os.path.join(validation_folder, "hole_cell", "h1_f32_dt0.01.json")))[
    "runs"
]
tvb_cell = [
    json.load(open(p)) for p in sorted(glob.glob(os.path.join(tvb_folder, "hole_cell", "*.json")))
]
tvb_lost = sum(not r.get("finite", True) for r in tvb_cell)
T_repeat.append(["TVB (double precision), step 0.01", f"{tvb_lost} of {len(tvb_cell)}"])
mean_se = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
tvb_rate = mean_se([r["m1_mean_r"] for r in tvb_cell if r.get("finite", True)])
jax_rate = mean_se([r["m1_mean_r"] for r in jax_cell if r["finite"]])
half_step_also_full = set(lost_seeds["h3_f32_dt0.005.json"]) <= set(
    lost_seeds["h1_f32_dt0.01.json"]
)

# ---------------------------------------------------------------- figure 2: the last steps before one run fails
record = np.load(os.path.join(validation_folder, "blowup_record_f32.npz"))
failing = int(record["bad_region"])
r_fail, V_fail = record["r"][:, failing], record["V"][:, failing]
last = len(r_fail) - 1  # index of the first non-finite step
window = slice(last - 110, last - 3)
steps_before = np.arange(window.start, window.stop) - last
fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.2))
ax = axes[0]
ax.plot(steps_before, V_fail[window], color=C_F32, lw=1.8)
ax.set_yscale("log")
ax.set_xlabel("steps before the first non-finite value")
ax.set_ylabel("V of the region that fails (log scale)")
ax.set_title("Its potential climbs without turning")
ax = axes[1]
shown_rate = np.maximum(r_fail[window], 1e-3)
ax.plot(steps_before, shown_rate, color=C_F32, lw=1.8)
at_bound = r_fail[window] == 0
ax.scatter(
    steps_before[at_bound],
    shown_rate[at_bound],
    s=28,
    color=INK,
    zorder=4,
    label="rate at the zero bound",
)
ax.set_yscale("log")
ax.set_xlabel("steps before the first non-finite value")
ax.set_ylabel("r of the same region (log scale)")
ax.set_title("Its rate stays near zero until the last steps")
ax.legend(loc="upper left", fontsize=8)
save(fig, "g2_before_failure.png")
rank_of_failing = int((record["in_strength"] > record["in_strength"][failing]).sum()) + 1
steps_at_bound = int(at_bound.sum())
unusual_before = int(((record["V"][: last - 200] > 2) & (record["r"][: last - 200] < 0.3)).sum())
V_shown = ", ".join(f"{V_fail[last + k]:.0f}" for k in (-30, -20, -14, -10, -8, -6, -5, -4))

# ---------------------------------------------------------------- the note
blocks = [
    ("h1", "What the grey cells in the maps are"),
    (
        "meta",
        "Draft, machine-written, 2026-11-25. A companion to `doc/sweep_writeup.pdf`. Every number is computed from the "
        "run files by `src/sweep/writeup_grey_cells.py`. The checks were written down in `doc/LOG.md` before they were "
        "run (entry of 2026-11-24, 21:22).",
    ),
    ("h2", "Two kinds of grey"),
    (
        "p",
        f"The cosine maps of the sweep write-up cover {category.size} settings of coupling G and noise σ. "
        f"{n_used} are used. The other {n_quiet + n_holes} are drawn grey, for two unrelated reasons.",
    ),
    (
        "fig",
        "g1_where.png",
        "Left: every cell of the map, coloured by why it is or is not used. Middle: how many of a cell's 28 runs at "
        "step 0.01 ended with non-finite values (blank: none). Right: the largest firing rate any finite run in the "
        "cell reached, as an average over one output sample. Cross: the paper's working point.",
    ),
    (
        "ul",
        [
            f"**Unresponsive cells ({n_quiet}).** The network sits in the down state and changing G does almost nothing to the "
            "statistics, so there is no effect of G to compare the other parameters with.",
            f"**Cells with diverged runs ({n_holes}).** At least one run there, or in the next cell along G, ended with "
            "non-finite values. The registered rule was to leave such cells out, and their neighbours cannot have a "
            "derivative in G.",
        ],
    ),
    ("h2", "1. The unresponsive cells"),
    (
        "ul",
        [
            f"{int(quiet_corner.sum())} of the {n_quiet} form one block at low G (G ≤ 0.53). In them the share of time in "
            f"the up state is under {100 * np.nanmax(up_occupancy[quiet_corner]):.1f}% and the mean rate is "
            f"{np.nanmin(mean[RATE][quiet_corner]):.2f} to {np.nanmax(mean[RATE][quiet_corner]):.2f}.",
            f"Example, G 0.45 and σ 0.022: one grid increment of G changes the mean rate by {example_change:+.5f} "
            f"± {example_se:.5f}. At the working point the same step changes it by "
            f"{G_effect[RATE][working_point] * 0.01:+.3f}.",
            "The rule was that G's combined effect on the seven statistics must exceed three standard errors before "
            "a cosine is computed. In these cells it does not. Nothing fails there; there is nothing to compare.",
            f"The other {n_quiet - int(quiet_corner.sum())} unresponsive cells are scattered single cells, mostly in the "
            "lowest-noise row, where the estimate did not clear the same rule.",
        ],
    ),
    ("h2", "2. Runs that end with non-finite values"),
    (
        "p",
        f"**How often and where.** {n_lost} of the {n_made:,} runs at time step 0.01 diverged (ended with non-finite values), and {half_lost} of the "
        f"{len(half_step):,} at half the time step. None diverged below σ {first_sigma_with_loss:.3f} or below G "
        f"{lowest_G_with_loss:.2f}. The largest rates reached by the runs that stay finite grow towards the same "
        "corner (Figure 1, right).",
    ),
    ("table", ["σ", "Runs that diverged at time step 0.01 (all G)"], T_sigma),
    (
        "p",
        f"**When in the run.** Not at the start. The diverged runs first show a non-finite value at volume "
        f"{int(np.percentile(failure_volume, 25))}, {int(np.median(failure_volume))} and "
        f"{int(np.percentile(failure_volume, 75))} of 833 (quartiles); {int((failure_volume < 50).sum())} of "
        f"{len(failure_volume)} fail in the first 50 volumes.",
    ),
    (
        "p",
        f"**What sets it off.** Runs at different settings that share a seed share their noise. Several of them fail "
        f"at the same volume: the {len(failure_volume)} diverged runs are {len(events)} distinct events. So a particular "
        "stretch of noise triggers the failure, across a range of settings.",
    ),
    ("table", ["Seed", "Volume", "Settings that fail together", "G", "σ"], T_events),
    (
        "p",
        f"**What happens.** One failure was recorded step by step (G 0.68, σ 0.054, single precision, seed "
        f"{int(record['seed'])}, at {float(record['time_units']):.0f} time units). The first non-finite value appears "
        f"in the most strongly connected region (rank {rank_of_failing} of 84 by in-strength).",
    ),
    (
        "fig",
        "g2_before_failure.png",
        "The region where the first non-finite value appears, over the last steps before it. Left: its potential V. "
        "Right: its rate r; black dots mark steps at which the rate sits at the zero bound. The last three steps "
        "before the failure are off the scale and not drawn.",
    ),
    (
        "ul",
        [
            f"About 80 steps before the end the region's rate is knocked to the zero bound ({steps_at_bound} steps at "
            "the bound in the window shown) while its potential is positive and rising.",
            "In the model, a positive potential normally makes the rate grow, and the growing rate then pulls the "
            "potential back down. With the rate near zero that brake is missing.",
            f"The potential climbs without turning: {V_shown}. Only in the last eight steps does the rate follow.",
            "The spike is then too fast for the time step: the next value overshoots to a large negative potential with "
            "the rate at zero, and two steps later the values are infinite.",
            f'The state "potential above 2 with rate below 0.3" occurs {unusual_before} times in the '
            f"{last - 200:,} earlier steps of that run, in any region.",
        ],
    ),
    (
        "p",
        "**What it is not.** The same setting was run again, 64 new seeds per variant, and in TVB.",
    ),
    ("table", ["Repeat runs at G 0.68, σ 0.054", "Runs that diverged"], T_repeat),
    (
        "ul",
        [
            "**Not the JAX code.** TVB loses runs at this setting too. The mean rate of the runs that stay finite is "
            f"{tvb_rate[0]:.4f} ± {tvb_rate[1]:.4f} in TVB and {jax_rate[0]:.4f} ± {jax_rate[1]:.4f} in JAX.",
            "**Not single precision.** Double precision loses as many.",
            "**Not simply the time-step size.** At half the time step fewer runs diverge, but the difference is too small for this "
            "many runs to resolve."
            + (
                " The seeds that diverge at half the time step are all among those that diverge at the full time step, at different times; "
                "that is unexplained."
                if half_step_also_full
                else ""
            ),
        ],
    ),
    ("h2", "What is and is not established"),
    (
        "ul",
        [
            "**Established:** where the losses are, how often, that they share noise events, that they occur in TVB and "
            "in both precisions, and what one of them looks like step by step.",
            "**An interpretation of one recorded case:** that the losses come from additive noise acting on a rate "
            "held at a zero bound, in a strongly driven region. A hand estimate made before the record was taken "
            "(a growing oscillation at high rate) was wrong.",
            "**Not established:** whether the zero bound is necessary; how the frequency of divergence depends on the time step; "
            "whether the spike would stay finite in the exact equations.",
            "**For the paper's working point** (σ 0.030) none of this applies: no run diverged at or below σ 0.038.",
        ],
    ),
    (
        "meta",
        "Full record: `doc/LOG.md`. Figures: `results/figures/grey_cells/`. Rebuild: `src/runall.sh`.",
    ),
]
build_pdf(blocks, figures, out_path)
