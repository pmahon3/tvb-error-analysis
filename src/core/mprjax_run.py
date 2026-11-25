"""Run one batch of simulations with the JAX kernel (mprjax.py) and write their summary statistics.

Usage:
    python3 mprjax_run.py <job.json> <output_prefix>

The job file describes the batch:
    {"conn": "hcp001_conn.npz", "speed": 20.0, "dt": 0.01,
     "params": [[G, eta, J, Delta, sigma], ...],      one row per parameter setting
     "seeds": [...],                                  every setting is run with every seed
     "T_units": 60000, "warm_units": 1000,            length kept, and length discarded at the start
     "n_tr_block": 7, "tag": "...", "save_bold": true}

It writes
    <output_prefix>.json   one record per run: the parameters and the summary statistics, with the same field names
                           as the TVB reference runs (src/tvb_reference/tvb_ref_run.py)
    <output_prefix>.npz    arrays per run: BOLD per volume, share of regions "up" per volume, per-region means

Column b of the batch is setting b % (number of settings) with seed b // (number of settings).
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

import json
import time

import jax
import numpy as np

import mprjax
from mprjax import SAMPLES_PER_UNIT, UNITS_PER_TR
from repo_paths import data

SAMPLES_PER_VOLUME = UNITS_PER_TR * SAMPLES_PER_UNIT
BOLD_VOLUMES_DROPPED = int(30 / 0.72)  # the first 30 s of BOLD are left out of the BOLD statistics
MIN_VOLUMES_FOR_BOLD_STATISTICS = 50


def host_stats(bold, sum_r, up7, up5, mx, sx, sxx, W, n_tr):
    """Summary statistics of one run, from what the device accumulated.

    bold           (volumes, regions) BOLD at the end of each volume
    sum_r          (volumes, regions) sum of the rate over each volume's 720 output samples
    up7, up5       (volumes, regions) number of those samples with rate above 0.7, above 0.5
    mx             (volumes,) largest rate in each volume
    sx, sxx        sum of x and of x x^T over the run, x = rate at the last sample of each time unit
    W              connectome weights
    n_tr           number of volumes

    Returns (statistics dict, share of region-samples above 0.7 per volume, the same above 0.5).
    """
    from stats import dfc_w, switching_index, vaf2

    n_regions = bold.shape[1]
    n_samples = n_tr * SAMPLES_PER_VOLUME
    n_units = n_tr * UNITS_PER_TR
    sum_r, up7, up5 = (np.asarray(a, np.float64) for a in (sum_r, up7, up5))

    # share of regions in the high state, per volume
    up_fraction_07 = up7.sum(1) / (SAMPLES_PER_VOLUME * n_regions)
    up_fraction_05 = up5.sum(1) / (SAMPLES_PER_VOLUME * n_regions)

    # covariance of the rates across regions, and its eigenvalues
    covariance = (sxx - np.outer(sx, sx) / n_units) / (n_units - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]

    statistics = dict(
        m1_mean_r=float(sum_r.sum() / (n_samples * n_regions)),
        m2_occ_up07=float(up7.sum() / (n_samples * n_regions)),
        occ_up05=float(up5.sum() / (n_samples * n_regions)),
        m3_u_median=float(np.median(up_fraction_07)),
        m4_u_p05=float(np.percentile(up_fraction_07, 5)),
        u_p95=float(np.percentile(up_fraction_07, 95)),
        u_sd=float(up_fraction_07.std()),
        m5_vaf2_r=float(eigenvalues[:2].sum() / eigenvalues.sum())
        if eigenvalues.sum() > 0
        else float("nan"),
        max_r=float(mx.max()),
    )

    # statistics of the BOLD signal; records too short for them (test runs) get none
    kept_bold = bold[BOLD_VOLUMES_DROPPED:].astype(np.float64)
    upper_triangle = np.triu_indices(n_regions, 1)
    if kept_bold.shape[0] >= MIN_VOLUMES_FOR_BOLD_STATISTICS and np.isfinite(kept_bold).all():
        functional_connectivity = np.nan_to_num(np.corrcoef(kept_bold.T))
        window_correlations, windows = dfc_w(kept_bold)
        statistics.update(
            switching_index=float(switching_index(window_correlations, windows)),
            mean_fc=float(functional_connectivity[upper_triangle].mean()),
            corr_fc_sc=float(
                np.corrcoef(functional_connectivity[upper_triangle], (W + W.T)[upper_triangle])[
                    0, 1
                ]
            ),
            vaf2_bold=vaf2(kept_bold),
        )
    return statistics, up_fraction_07, up_fraction_05


def main():
    job = json.load(open(sys.argv[1]))
    output_prefix = sys.argv[2]

    connectome = np.load(data(job["conn"]))
    W, tract_lengths = connectome["weights"], connectome["tract_lengths"]
    params = np.asarray(job["params"], float)
    seeds = list(job["seeds"])
    dt = float(job["dt"])
    volumes_per_block = int(job.get("n_tr_block", 7))
    n_volumes = int(job["T_units"] // UNITS_PER_TR)
    n_volumes -= n_volumes % volumes_per_block
    warm_units = int(job["warm_units"])

    print("devices:", jax.devices(), "| x64:", bool(jax.config.jax_enable_x64), flush=True)
    runner = mprjax.build(
        W, tract_lengths, float(job["speed"]), dt, params, seeds, n_tr_block=volumes_per_block
    )
    batch, n_regions = runner["B"], runner["N"]
    total_steps = (
        (warm_units + n_volumes * UNITS_PER_TR) * SAMPLES_PER_UNIT * runner["steps_per_sample"]
    )
    print(
        f"B = {batch} ({runner['C']} configs x {runner['S']} seeds); N = {n_regions}; H = {runner['H']}; "
        f"dt = {dt}; {n_volumes} TRs in blocks of {volumes_per_block}; {total_steps} steps",
        flush=True,
    )

    # discard the start of the run
    started = time.time()
    state = runner["init"]()
    state = runner["warm"](state, np.int32(0), np.arange(warm_units, dtype=np.int32))
    jax.block_until_ready(state[0])
    warm_seconds = time.time() - started
    print(f"warm-up ({warm_units} units, with compilation): {warm_seconds:.1f} s", flush=True)

    # the kept part, a block of volumes at a time; what each block accumulated is copied to the host
    hemo = runner["fresh_hemo"]()
    single = (
        np.float32
    )  # per-volume sums have at most 720 terms: single precision holds them exactly enough
    bold = np.zeros((n_volumes, batch, n_regions), single)
    sum_r = np.zeros((n_volumes, batch, n_regions), single)
    up7 = np.zeros((n_volumes, batch, n_regions), single)
    up5 = np.zeros((n_volumes, batch, n_regions), single)
    largest_rate = np.zeros((n_volumes, batch))
    sum_x = np.zeros((batch, n_regions))
    sum_xx = np.zeros((batch, n_regions, n_regions))
    finite = np.ones(batch, bool)
    block_seconds = []
    n_blocks = n_volumes // volumes_per_block
    for k in range(n_blocks):
        block_started = time.time()
        first_unit = np.int32(warm_units + k * volumes_per_block * UNITS_PER_TR)
        state, hemo, per_volume, block_sum_x, block_sum_xx, block_finite = runner["block"](
            state, hemo, first_unit
        )
        volumes = slice(k * volumes_per_block, (k + 1) * volumes_per_block)
        # on the device the layout is (volumes, regions, batch); here it is (volumes, batch, regions)
        block_bold, block_sum_r, block_up7, block_up5, block_largest = (
            np.asarray(a) for a in per_volume
        )
        bold[volumes] = block_bold.transpose(0, 2, 1)
        sum_r[volumes] = block_sum_r.transpose(0, 2, 1)
        up7[volumes] = block_up7.transpose(0, 2, 1)
        up5[volumes] = block_up5.transpose(0, 2, 1)
        largest_rate[volumes] = block_largest
        sum_x += np.asarray(block_sum_x, np.float64).T
        sum_xx += np.asarray(block_sum_xx, np.float64)
        finite &= np.asarray(block_finite)
        block_seconds.append(time.time() - block_started)
        if k in (0, 1, 2) or (k + 1) % 20 == 0:
            print(
                f"block {k + 1}/{n_blocks}: {block_seconds[-1]:.2f} s; "
                f"finite rows {int(finite.sum())}/{batch}",
                flush=True,
            )

    # timing; the first block includes compilation, so it is left out of the median
    wall_seconds = time.time() - started
    steps_per_block = (
        volumes_per_block * UNITS_PER_TR * SAMPLES_PER_UNIT * runner["steps_per_sample"]
    )
    typical_block = (
        float(np.median(block_seconds[1:])) if len(block_seconds) > 1 else block_seconds[0]
    )
    timing = dict(
        wall_s=round(wall_seconds, 1),
        warm_s=round(warm_seconds, 1),
        median_block_s=round(typical_block, 3),
        us_per_step=round(typical_block / steps_per_block * 1e6, 2),
        us_per_step_per_run=round(typical_block / steps_per_block * 1e6 / batch, 4),
        s_per_run=round(wall_seconds / batch, 2),
        B=batch,
    )
    print("timing:", json.dumps(timing), flush=True)

    # one record per run
    records = []
    up_fraction_07 = np.zeros((batch, n_volumes), np.float32)
    up_fraction_05 = np.zeros((batch, n_volumes), np.float32)
    for column in range(batch):
        setting = int(runner["config_of"][column])
        G, eta, J, Delta, sigma = (float(x) for x in params[setting])
        record = dict(
            G=G,
            eta=eta,
            J=J,
            Delta=Delta,
            nsig=sigma,
            dt=dt,
            seed=int(seeds[int(runner["seed_of"][column])]),
            config=setting,
            T=float(n_volumes * UNITS_PER_TR),
            warm=float(warm_units),
            finite=bool(finite[column]) and bool(np.isfinite(bold[:, column]).all()),
            wall_s=timing["s_per_run"],
            n_regions=n_regions,
            horizon=int(runner["H"]),
            backend="mprjax",
            x64=bool(jax.config.jax_enable_x64),
            tag=job.get("tag", ""),
        )
        if record["finite"]:
            statistics, up_fraction_07[column], up_fraction_05[column] = host_stats(
                bold[:, column],
                sum_r[:, column],
                up7[:, column],
                up5[:, column],
                largest_rate[:, column],
                sum_x[column],
                sum_xx[column],
                W,
                n_volumes,
            )
            record.update(statistics)
        records.append(record)

    # the statistics are written first; the arrays are large
    json.dump(
        dict(job=job, timing=timing, runs=records), open(output_prefix + ".json", "w"), indent=1
    )
    n_samples = n_volumes * SAMPLES_PER_VOLUME
    arrays = dict(
        u07=up_fraction_07,
        u05=up_fraction_05,
        node_occ07=(up7.sum(0, dtype=np.float64) / n_samples).astype(np.float32),
        node_mean_r=(sum_r.sum(0, dtype=np.float64) / n_samples).astype(np.float32),
        params=params,
        seeds=np.array(seeds),
        dt=dt,
    )
    if job.get("save_bold", True):
        arrays["bold"] = np.transpose(bold, (1, 0, 2))
    np.savez(output_prefix + ".npz", **arrays)
    print("saved", output_prefix + ".json", flush=True)


if __name__ == "__main__":
    main()
