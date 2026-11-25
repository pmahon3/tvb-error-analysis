"""What happens just before a run goes non-finite? (doc/LOG.md, 2026-11-24, 21:25, part B.)

Runs many seeds at one parameter setting with the JAX kernel, stops at the first non-finite value and keeps the last
steps of the run that failed.
Usage (from the repository root):
    src/.venv_vbjax/bin/python -I src/validation/blowup_record.py <G> <sigma> <dt> <first_seed> <n_seeds> <out.npz> [x64]
"""

import os
import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

G, sigma, dt = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
first_seed, n_seeds, out_path = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
if len(sys.argv) > 7 and sys.argv[7] == "x64":
    os.environ["JAX_ENABLE_X64"] = "1"

import time

import jax
import jax.numpy as jnp
import numpy as np

import mprjax
from repo_paths import data

ETA, J, DELTA = -5.0, 15.0, 1.0
T_MAX_UNITS = 61000
CHUNKS_PER_CALL = 200  # one call advances 200 output samples

conn = np.load(data("hcp001_conn.npz"))
W, L = conn["weights"], conn["tract_lengths"]
N = W.shape[0]
seeds = list(range(first_seed, first_seed + n_seeds))
B = len(seeds)
FT = mprjax.ftype()
steps_per_sample = int(round(mprjax.TAVG_PERIOD / dt))
steps_per_call = CHUNKS_PER_CALL * steps_per_sample
delays = mprjax.delays(L, 20.0, dt)
one = np.ones((1, B))
params = dict(G=G * one, eta=ETA * one, J=J * one, Delta=DELTA * one)
sample, gather, H = mprjax.make_sample(W, delays, dt, params, steps_per_sample)

# starting state and noise exactly as mprjax.build does
history_length = int(delays.max()) + 1
r_history = np.stack(
    [np.random.default_rng(1000 * s + 11).uniform(0.0, 2.0, (history_length, N)) for s in seeds]
)
V_start = np.stack([np.random.default_rng(1000 * s + 12).uniform(-2.0, 1.5, N) for s in seeds])
state = mprjax.start_state(np.transpose(r_history, (1, 2, 0)), V_start.T, gather, H)
keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
noise_r, noise_V = np.sqrt(2 * sigma * dt), np.sqrt(4 * sigma * dt)


@jax.jit
def advance(state, first_sample):
    def one_sample(state, sample_index):
        z = jax.vmap(
            lambda key: jax.random.normal(
                jax.random.fold_in(key, sample_index), (steps_per_sample, 2, N), FT
            )
        )(keys)
        z = jnp.transpose(z, (1, 2, 3, 0))  # (steps, 2, regions, runs)
        state, (r, V) = sample(state, (FT(noise_r) * z[:, 0], FT(noise_V) * z[:, 1]))
        return state, (r, V)

    state, (r, V) = jax.lax.scan(one_sample, state, first_sample + jnp.arange(CHUNKS_PER_CALL))
    return state, r.reshape(steps_per_call, N, B), V.reshape(steps_per_call, N, B)


print(f"G {G}, sigma {sigma}, dt {dt}, {B} seeds from {first_seed}, {FT.__name__}", flush=True)
previous = None
largest_rate_seen = np.zeros(B)
started = time.time()
n_calls = int(T_MAX_UNITS * mprjax.SAMPLES_PER_UNIT / CHUNKS_PER_CALL)
for call in range(n_calls):
    state, r, V = advance(state, np.int32(call * CHUNKS_PER_CALL))
    r, V = np.asarray(r), np.asarray(V)
    finite = np.isfinite(r).all((0, 1)) & np.isfinite(V).all((0, 1))
    if not finite.all():
        run = int(np.argmax(~finite))
        r_run = np.concatenate([previous[0][:, :, run], r[:, :, run]]) if previous else r[:, :, run]
        V_run = np.concatenate([previous[1][:, :, run], V[:, :, run]]) if previous else V[:, :, run]
        bad_step = int(np.argmax(~(np.isfinite(r_run).all(1) & np.isfinite(V_run).all(1))))
        bad_region = int(np.argmax(~(np.isfinite(r_run[bad_step]) & np.isfinite(V_run[bad_step]))))
        time_units = (call * steps_per_call + bad_step - (len(r_run) - steps_per_call)) * dt
        np.savez_compressed(
            out_path,
            r=r_run[: bad_step + 1],
            V=V_run[: bad_step + 1],
            G=G,
            sigma=sigma,
            dt=dt,
            seed=seeds[run],
            bad_region=bad_region,
            time_units=time_units,
            in_strength=W.sum(1),
            largest_rate_seen_before=largest_rate_seen,
        )
        print(
            f"first non-finite value: seed {seeds[run]}, region {bad_region}, at {time_units:.1f} time units",
            flush=True,
        )
        print(f"saved the last {bad_step + 1} steps of that run to {out_path}")
        break
    largest_rate_seen = np.maximum(largest_rate_seen, r.max((0, 1)))
    previous = (r, V)
    if (call + 1) % 250 == 0:
        units = (call + 1) * steps_per_call * dt
        print(
            f"{units:.0f} time units, all finite; largest rate so far {largest_rate_seen.max():.2f}; {time.time() - started:.0f} s",
            flush=True,
        )
else:
    np.savez_compressed(
        out_path, none=True, largest_rate_seen_before=largest_rate_seen, G=G, sigma=sigma, dt=dt
    )
    print(
        f"no non-finite value in {T_MAX_UNITS} time units; largest rate per run saved to {out_path}"
    )
