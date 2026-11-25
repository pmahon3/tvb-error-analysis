"""One uncoupled MPR node at constant input: how the scheme's statistics depend on dt, with and without the r >= 0 clamp.
Registered in doc/LOG.md (2026-11-24, "why the statistics move with dt"). Same drift and scheme as ridge_v2_sde.py:
Heun drift, Euler noise sigma*sqrt(dt)*N(0,1), r <- max(r, 0). Variants: 'clamp' (as used), 'free' (no clamp),
'vonly' (noise on v only; clamp kept, expected never to bind). float64.
Usage: .venv_vbjax/bin/python -I ridge_v2_node_dt.py <out.json> [replicates] [T_ms]
"""

import sys
from pathlib import Path

# make this folder and the shared modules in src/common importable (the scripts are run with python -I)
sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "common")]

import sys, json, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import vbjax as vb

ETA, SIG, BURN_MS = -5.0, 0.30, 200
R_DN, V_DN, R_UP, V_UP = 0.081, -1.962, 1.031, -0.154
DTS = [0.05, 0.025, 0.0125, 0.00625]
INPUTS = [0.2, 0.65]
STARTS = ["down", "up"]
VARIANTS = ["clamp", "free", "vonly"]
p0 = vb.mpr_default_theta


def drift(r, v, I):
    dr = (p0.Delta / (jnp.pi * p0.tau) + 2 * r * v) / p0.tau
    dv = (v ** 2 + ETA + p0.J * p0.tau * r + I - (jnp.pi ** 2) * (r ** 2) * (p0.tau ** 2)) / p0.tau
    return dr, dv


def build(dt, variant, R, T_ms):
    spm = int(round(1.0 / dt)); C = len(INPUTS) * len(STARTS); B = C * R
    I = jnp.asarray(np.repeat(np.repeat(INPUTS, len(STARTS)), R))                 # cell = input-major, then start
    up0 = np.tile(np.repeat([s == "up" for s in STARTS], R), len(INPUTS))
    r0 = jnp.asarray(np.where(up0, R_UP, R_DN)); v0 = jnp.asarray(np.where(up0, V_UP, V_DN))
    sr = 0.0 if variant == "vonly" else SIG
    clamp = variant != "free"
    key0 = jax.random.PRNGKey(20261124)                                          # same noise for all variants at a given dt

    def step(carry, z):
        r, v, acc, w = carry
        dr, dv = drift(r, v, I); dr2, dv2 = drift(r + dt * dr, v + dt * dv, I)
        rn = r + 0.5 * dt * (dr + dr2) + sr * jnp.sqrt(dt) * z[0]
        v = v + 0.5 * dt * (dv + dv2) + SIG * jnp.sqrt(dt) * z[1]
        bind = (rn < 0).astype(jnp.float64)
        r = jnp.maximum(rn, 0.0) if clamp else rn
        dn = (r < 0.5).astype(jnp.float64)
        s_bind, s_r, s_rdn, s_dn, s_up, mx = acc
        acc = (s_bind + w * bind, s_r + w * r, s_rdn + w * r * dn, s_dn + w * dn, s_up + w * (1 - dn), jnp.maximum(mx, w * r))
        return (r, v, acc, w), None

    def one_ms(carry, ms):
        r, v, acc = carry
        z = jax.random.normal(jax.random.fold_in(key0, ms), (spm, 2, B))
        w = (ms >= BURN_MS).astype(jnp.float64)
        (r, v, acc, _), _ = jax.lax.scan(step, (r, v, acc, w), z)
        return (r, v, acc), None

    @jax.jit
    def run():
        z0 = jnp.zeros(B)
        (r, v, acc), _ = jax.lax.scan(one_ms, (r0, v0, (z0, z0, z0, z0, z0, z0)), jnp.arange(T_ms))
        return r, v, acc

    return run, spm, C


def main():
    out = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 256; T_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    rows = []
    for dt in DTS:
        for variant in VARIANTS:
            t0 = time.time(); run, spm, C = build(dt, variant, R, T_ms)
            r, v, acc = run(); n = (T_ms - BURN_MS) * spm
            s_bind, s_r, s_rdn, s_dn, s_up, mx = [np.asarray(a).reshape(C, R) for a in acc]
            fin = (np.isfinite(np.asarray(r)) & np.isfinite(np.asarray(v))).reshape(C, R)
            for ci in range(C):
                I, start = INPUTS[ci // len(STARTS)], STARTS[ci % len(STARTS)]
                ok = fin[ci]; m = int(ok.sum())
                def ms(x):                                                       # mean and se over finite replicates
                    x = x[ok]; return (float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (float("nan"), float("nan"))
                with np.errstate(invalid="ignore", divide="ignore"):
                    rdn = s_rdn[ci] / s_dn[ci]
                okd = ok & (s_dn[ci] > 0.5 * n)                                  # replicates mostly down: conditional mean is meaningful
                rows.append(dict(dt=dt, variant=variant, I=I, start=start, finite=m, R=R,
                                 bind=ms(s_bind[ci] / n), mean_r=ms(s_r[ci] / n), occ_up=ms(s_up[ci] / n),
                                 r_down=(float(rdn[okd].mean()), float(rdn[okd].std(ddof=1) / np.sqrt(okd.sum()))) if okd.sum() > 1 else (float("nan"), float("nan")),
                                 n_mostly_down=int(okd.sum()), max_r=float(np.nanmax(np.where(ok, mx[ci], np.nan))) if m else float("nan")))
            print(f"dt={dt} {variant}: {time.time() - t0:.1f} s", flush=True)
    json.dump(rows, open(out, "w"), indent=1)
    for variant in VARIANTS:
        print(f"\n{variant}")
        for I in INPUTS:
            for start in STARTS:
                print(f"  I={I} start={start}")
                for row in rows:
                    if row["variant"] == variant and row["I"] == I and row["start"] == start:
                        print(f"    dt={row['dt']:<8} finite {row['finite']:3d}/{row['R']}  bind {row['bind'][0]:.4f}  mean_r {row['mean_r'][0]:.4f}±{row['mean_r'][1]:.4f}"
                              f"  r_down {row['r_down'][0]:.5f}±{row['r_down'][1]:.5f} (n={row['n_mostly_down']})  occ_up {row['occ_up'][0]:.4f}±{row['occ_up'][1]:.4f}  max_r {row['max_r']:.2f}")


if __name__ == "__main__":
    main()
