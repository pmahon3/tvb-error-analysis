"""Verify what the installed TVB actually runs for the companion pipeline (ins-amu/PCI_Fluidity), before any long run.
Checks: installed versions; hashes of the installed backend files (to compare with the master copies read on
2026-11-24, kept in external/tvb_master/); the rendered integration kernel (is the bound applied after both Heun
stages?); the per-variable sd of the prefilled noise buffer; that r lands exactly on 0 in a short raw run; timing.
Configuration is OFF the paper's working point on purpose (G = 0.30, nsig = 0.040): timing and finiteness only.
Usage (tower, venv_tvb): python tvb_ref_verify.py
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import sys, os, time, hashlib
HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "pci_ref")
sys.path.insert(0, os.path.join(REF, "utils"))
import numpy as np
from importlib.metadata import version
import numba

print("versions: tvb-library", version("tvb-library"), "| numpy", np.__version__, "| numba", numba.__version__, "| python", sys.version.split()[0], flush=True)
import tvb.simulator.backend.nb_mpr as nbm
import tvb.simulator.backend.templates as tpl
import tvb.simulator.noise as tnoise
import tvb.simulator.integrators as tint
import tvb.simulator.models.infinite_theta as tinf
for f in [nbm.__file__, os.path.join(os.path.dirname(tpl.__file__), "nb-montbrio.py.mako"), tnoise.__file__, tint.__file__, tinf.__file__]:
    raw = open(f, "rb").read().replace(b"\r\n", b"\n")
    print("  sha256[:16]", hashlib.sha256(raw).hexdigest()[:16], os.path.basename(f), len(raw), "bytes")

import simulation                                   # the companion repository's utils/simulation.py, unchanged
G, NSIG, DT = 0.30, 0.040, 0.01
conn = simulation.configure_conn(os.path.join(REF, "hcp-001.zip"), conn_speed=20.)
sim = simulation.configure_sim(G=G, nsigma=NSIG, conn=conn, dt=DT, seed=42)
N = conn.number_of_regions
nz = conn.weights > 0
print(f"connectome: N = {N}; horizon = {conn.horizon}; idelays over weighted links: min {conn.idelays[nz].min()}, max {conn.idelays[nz].max()}; max weight {conn.weights.max():.3f}")
print("model:", type(sim.model).__name__, "eta", sim.model.eta, "J", sim.model.J, "Delta", sim.model.Delta, "tau", sim.model.tau,
      "| boundaries:", sim.model.state_variable_boundaries, "| coupling:", type(sim.coupling).__name__, "a =", sim.coupling.a)
print("integrator:", type(sim.integrator).__name__, "dt", sim.integrator.dt, "| nsig shape", sim.integrator.noise.nsig.shape,
      "values", np.unique(sim.integrator.noise.nsig).tolist(), "| gfun shape", np.shape(sim.integrator.noise.gfun(None)))

backend = nbm.NbMPRBackend()
src = backend.render_template('<%include file="nb-montbrio.py.mako"/>', dict(sim=sim, compatibility_mode=False))
lines = [l for l in src.split("\n") if l.strip()]
i0 = next(i for i, l in enumerate(lines) if "def integrate" in l)
print("--- rendered kernel: bound functions and the Heun step ---")
for l in lines:
    if l.startswith("def bound_") or ("x = x if" in l): print("   ", l.rstrip())
for l in lines[i0:]:
    s = l.strip()
    if any(k in s for k in ("_noise =", "_int =", "_n =", "bound_", "cx(")) and not s.startswith("#"): print("   ", s[:150])

gf = sim.integrator.noise.gfun(None)
buf = sim.integrator.noise.generate(shape=(sim.model.nvar, N, 50000)) * gf
print(f"noise buffer sd per variable: r {buf[0].std():.6f} (expected sqrt(2*nsig*dt) = {np.sqrt(2 * NSIG * DT):.6f}); "
      f"V {buf[1].std():.6f} (expected sqrt(4*nsig*dt) = {np.sqrt(4 * NSIG * DT):.6f}); sd along nodes of the per-node sd: r {buf[0].std(1).std():.2e}, V {buf[1].std(1).std():.2e}")

t0 = time.time(); r, V = backend._run_sim_plain(sim, nstep=20000, print_source=False); t1 = time.time()
print(f"raw run, 20000 steps (200 time units): {t1 - t0:.1f} s including compilation; finite: {bool(np.isfinite(r).all() and np.isfinite(V).all())}; "
      f"min r = {r.min():.3g}; share of (node, step) with r exactly 0: {np.mean(r == 0):.4f}; share with r < 0: {np.mean(r < 0):.4f}")
t0 = time.time(); r, V = backend._run_sim_plain(sim, nstep=100000, print_source=False); t1 = time.time()
print(f"raw run, 100000 steps: {t1 - t0:.1f} s  ->  {(t1 - t0) / 100000 * 1e6:.1f} microseconds per step; finite: {bool(np.isfinite(r).all())}")
t0 = time.time(); (tt, dd), = simulation.run_nbMPR_backend(sim, simulation_length=2000); t1 = time.time()
print(f"their chunked path, 2000 time units: {t1 - t0:.1f} s; output shape {dd.shape}; time step of output {tt[1] - tt[0]:.3f}; finite: {bool(np.isfinite(dd).all())}")
