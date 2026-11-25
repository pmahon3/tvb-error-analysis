"""One run of the companion pipeline (ins-amu/PCI_Fluidity, utils/simulation.py, unchanged) and its statistics.
Registered in doc/LOG.md (2026-11-24, stage A'). The simulation is theirs; only the statistics below are this folder's.
Usage (tower, venv_tvb): python tvb_ref_run.py <G> <nsig> <dt> <seed> <out_prefix> [T_units=60000] [warm_units=1000]
Writes <out_prefix>.json (statistics) and <out_prefix>.npz (BOLD per TR, network up-fraction per TR, per-node values).
Conventions: their temporal average has period 0.1 time units; one sample is read as 1 ms (their "time x 10"), so
TR = 720 samples = 72 time units. chunksize is set to 1000 time units so the length divides evenly at every dt.
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
REF = (
    os.path.join(HERE, "pci_ref")
    if os.path.isdir(os.path.join(HERE, "pci_ref"))
    else os.path.join(HERE, "..", "..", "external", "pci_fluidity")
)
sys.path.insert(0, os.path.join(REF, "utils"))
sys.path.insert(0, HERE)
import numpy as np

PERIOD, TR_SAMPLES = 0.1, 720


def statistics(r, W):
    """r: (samples, N) rates after warm-up, one sample = 0.1 time units. W: the weights their code used."""
    from stats import dfc_w, switching_index, vaf2
    from mpr_net import Balloon

    N = r.shape[1]
    n_tr = r.shape[0] // TR_SAMPLES
    r = r[: n_tr * TR_SAMPLES]
    up7, up5 = r > 0.7, r > 0.5
    u7 = up7.reshape(n_tr, TR_SAMPLES, N).mean((1, 2))
    u5 = up5.reshape(n_tr, TR_SAMPLES, N).mean((1, 2))
    s = dict(
        m1_mean_r=float(r.mean()),
        m2_occ_up07=float(up7.mean()),
        occ_up05=float(up5.mean()),
        m3_u_median=float(np.median(u7)),
        m4_u_p05=float(np.percentile(u7, 5)),
        u_p95=float(np.percentile(u7, 95)),
        u_sd=float(u7.std()),
        m5_vaf2_r=vaf2(r[9::10]),
        max_r=float(r.max()),
    )
    hemo = Balloon(1, N)
    bold = np.zeros((n_tr, N), np.float32)
    for i in range(n_tr * TR_SAMPLES):
        hemo.step(r[i][None, :], 1e-3)
        if (i + 1) % TR_SAMPLES == 0:
            bold[(i + 1) // TR_SAMPLES - 1] = hemo.bold()[0]
    b = bold[int(30 / 0.72) :].astype(np.float64)
    iu = np.triu_indices(N, 1)
    if np.isfinite(b).all():
        FC = np.nan_to_num(np.corrcoef(b.T))
        D, Ws = dfc_w(b)
        s.update(
            switching_index=float(switching_index(D, Ws)),
            mean_fc=float(FC[iu].mean()),
            corr_fc_sc=float(np.corrcoef(FC[iu], (W + W.T)[iu])[0, 1]),
            vaf2_bold=vaf2(b),
        )
    arrays = dict(
        bold=bold,
        u07=u7.astype(np.float32),
        u05=u5.astype(np.float32),
        node_occ07=up7.mean(0).astype(np.float32),
        node_mean_r=r.mean(0).astype(np.float32),
    )
    return s, arrays


def main():
    G, nsig, dt, seed, out = (
        float(sys.argv[1]),
        float(sys.argv[2]),
        float(sys.argv[3]),
        int(sys.argv[4]),
        sys.argv[5],
    )
    T = float(sys.argv[6]) if len(sys.argv) > 6 else 60000.0
    warm = float(sys.argv[7]) if len(sys.argv) > 7 else 1000.0
    import simulation  # the companion repository's code

    conn = simulation.configure_conn(os.path.join(REF, "hcp-001.zip"), conn_speed=20.0)
    sim = simulation.configure_sim(G=G, nsigma=nsig, conn=conn, dt=dt, seed=seed)
    chunk = int(round(1000.0 / dt))
    t0 = time.time()
    ((tt, dd),) = simulation.run_nbMPR_backend(sim, simulation_length=T + warm, chunksize=chunk)
    wall = time.time() - t0
    res = dict(
        G=G,
        nsig=nsig,
        dt=dt,
        seed=seed,
        T=T,
        warm=warm,
        wall_s=round(wall, 1),
        n_regions=int(conn.number_of_regions),
        horizon=int(conn.horizon),
        finite=bool(np.isfinite(dd).all()),
        n_samples=int(dd.shape[0]),
        sample_period=float(tt[1] - tt[0]),
    )
    r = dd[int(round(warm / PERIOD)) :, 0, :, 0]
    if res["finite"]:
        s, arrays = statistics(r, np.asarray(conn.weights))
        res.update(s)
        np.savez_compressed(out + ".npz", **arrays)
    else:
        bad = np.where(~np.isfinite(dd[:, 0, :, 0]).all(1))[0]
        res["first_nonfinite_sample"] = int(bad[0]) if len(bad) else -1
    json.dump(res, open(out + ".json", "w"), indent=1)
    print(
        json.dumps({k: res[k] for k in ("G", "nsig", "dt", "seed", "finite", "wall_s")}), flush=True
    )


if __name__ == "__main__":
    main()
