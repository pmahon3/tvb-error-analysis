"""Smallest case for the per-region question (doc/LOG.md, 2026-11-24, 21:22, part A): three regions in a chain,
no noise. Does G move the middle region (two neighbours) more, relative to the end regions, than eta does?
Usage (from the repository root): JAX_ENABLE_X64=1 src/.venv_vbjax/bin/python -I src/validation/chain_response_check.py
"""

import os
import sys
from pathlib import Path

# make this folder and the shared modules in src/core importable (the scripts are run with python -I)
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "core"),
]

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np

import mprjax

weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
lengths = 20.0 * (weights > 0)  # one time unit of delay on each link
G0, ETA0, J0, DELTA0 = 0.5, -5.0, 15.0, 1.0
dG, dEta = 1e-3, 1e-3
settings = np.array(
    [
        [G0, ETA0, J0, DELTA0, 0.0],
        [G0 + dG, ETA0, J0, DELTA0, 0.0],
        [G0 - dG, ETA0, J0, DELTA0, 0.0],
        [G0, ETA0 + dEta, J0, DELTA0, 0.0],
        [G0, ETA0 - dEta, J0, DELTA0, 0.0],
    ]
)
run = mprjax.build(weights, lengths, 20.0, 0.01, settings, seeds=[1])
state = run["init"]()
for block in range(4):  # 4 x 500 time units, to let the network settle
    state = run["warm"](state, np.int32(500 * block), np.arange(500, dtype=np.int32))
    rates = np.asarray(state[0])  # (regions, settings)
    print(f"after {500 * (block + 1)} time units: base rates {np.round(rates[:, 0], 6).tolist()}")
to_G = (rates[:, 1] - rates[:, 2]) / (2 * dG)
to_eta = (rates[:, 3] - rates[:, 4]) / (2 * dEta)
print("response of each region's rate to G:  ", np.round(to_G, 6).tolist())
print("response of each region's rate to eta:", np.round(to_eta, 6).tolist())
print(f"middle / end, under G: {to_G[1] / to_G[0]:.4f}; under eta: {to_eta[1] / to_eta[0]:.4f}")
print(
    "coupling input sum_j W_ij r_j at the base state:", np.round(weights @ rates[:, 0], 6).tolist()
)
