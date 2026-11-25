"""Dump what the JAX port must reproduce exactly: the connectome as the companion pipeline configures it, and one short
raw run of TVB's kernel with the buffers captured at the kernel's entry (initial history and pre-filled noise).
Registered in doc/LOG.md (2026-11-24, port match). Usage (tower, venv_tvb): python tvb_ref_dump.py <out.npz> [nstep=3000]
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "common"),
]

import sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "pci_ref")
sys.path.insert(0, os.path.join(REF, "utils"))
import numpy as np
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import simulation


class Capture(NbMPRBackend):
    def _run_integrate(self, integrate, sim, N, nstep, svar_bufs, stimulus):
        self.captured = [b.copy() for b in svar_bufs]
        return super()._run_integrate(integrate, sim, N, nstep, svar_bufs, stimulus)


out = sys.argv[1]
nstep = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
G, NSIG, SPEED = 0.54, 0.030, 20.0
d = {}
for dt in (0.02, 0.01, 0.005, 0.0025):
    conn = simulation.configure_conn(os.path.join(REF, "hcp-001.zip"), conn_speed=SPEED)
    sim = simulation.configure_sim(G=G, nsigma=NSIG, conn=conn, dt=dt, seed=42)
    d[f"idelays_dt{dt}"] = np.asarray(sim.connectivity.idelays)
    d[f"horizon_dt{dt}"] = int(sim.connectivity.horizon)
    d[f"tavg_istep_dt{dt}"] = int(sim.monitors[0].istep)
    if dt == 0.01:
        b = Capture()
        r, V = b._run_sim_plain(sim, nstep=nstep, print_source=False)
        d.update(
            weights=np.asarray(sim.connectivity.weights),
            tract_lengths=np.asarray(sim.connectivity.tract_lengths),
            r_buf=b.captured[0],
            V_buf=b.captured[1],
            r_out=r,
            V_out=V,
            G=G,
            nsig=NSIG,
            speed=SPEED,
            dt=dt,
            eta=float(sim.model.eta[0]),
            J=float(sim.model.J[0]),
            Delta=float(sim.model.Delta[0]),
            tau=float(sim.model.tau[0]),
            cr=float(sim.model.cr[0]),
            cv=float(sim.model.cv[0]),
            I=float(sim.model.I[0]),
            coupling_b=float(sim.coupling.b[0]),
        )
        print(
            "captured buffers",
            b.captured[0].shape,
            "| output",
            r.shape,
            "| finite",
            bool(np.isfinite(r).all() and np.isfinite(V).all()),
            "| share of r exactly 0:",
            round(float(np.mean(r == 0)), 4),
            flush=True,
        )
np.savez_compressed(out, **d)
print(
    "saved",
    out,
    {
        k: (v.shape if hasattr(v, "shape") and getattr(v, "shape", ()) else v)
        for k, v in d.items()
        if not k.endswith("_buf") and not k.endswith("_out")
    },
)
