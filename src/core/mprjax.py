"""JAX port of TVB's numba MPR kernel (nb-montbrio.py.mako, default mode, tvb-library 2.10.0) for networks whose
weighted links all have a delay of at least one step. Registered in doc/LOG.md (2026-11-24, "JAX port").

One step i (state at column i from column i - 1), exactly as the template:
    c0[n] = G * sum_j W[n, j] * r_j(i - 1 - D[n, j])
    predictor: x + dt * f(x, c0) + noise;            r bounded at 0
    c1[n] = G * sum_j W[n, j] * r_j(i - D[n, j])
    result:    x + dt * (f(x, c0) + f(x_pred, c1)) / 2 + the same noise;   r bounded at 0
With D >= 1 on every weighted link, c0 of step i + 1 is c1 of step i, so the coupling is gathered once per step and
carried. (At D = 0 the template reads the column being written; that case is NOT this kernel. See doc/LOG.md, 10:20.)
Noise per step, as TVB's Additive noise with nsig = [sigma, 2 sigma]: sqrt(2 sigma dt) z on r, sqrt(4 sigma dt) z on V.
Batch rows carry their own (G, eta, J, Delta, sigma); rows that share a seed share the noise and the initial state.
"""
import numpy as np
import jax
import jax.numpy as jnp

TAU = 1.0
TAVG_PERIOD = 0.1  # their TemporalAverage period, in time units
SAMPLES_PER_UNIT = 10  # tavg samples per time unit
UNITS_PER_TR = 72  # TR = 720 tavg samples (their "time x 10": one sample read as 1 ms)
BALLOON = dict(kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32, E0=0.4, V0=0.04)  # as mpr_net.Balloon


def ftype():
    return np.float64 if jax.config.jax_enable_x64 else np.float32


def delays(L, speed, dt):
    return np.rint(L / speed / dt).astype(
        np.int32
    )  # TVB: idelays = rint(tract_lengths / speed / dt); checked against TVB


def make_sample(W, D, dt, P, S):
    """One chunk of S steps (one temporal-average sample). Batch is the LAST axis: r, V, c0 are (N, B); the ring buffer
    of r is (H, N, B) with column t in slot t mod H and H a multiple of S. Because every weighted delay is >= S steps,
    all delayed reads inside the chunk come from columns written before it: the buffer is constant inside the chunk
    and is updated once, with a slab (no per-step copy of the history; measured 20x faster on the GPU, same numbers).
    P: dict of (1, B) arrays G, eta, J, Delta. Returns sample(carry, noise), gather(buf, i), H.
    carry = (r, V, c0, buf, i) with i a multiple of S; noise = (nr, nV), each (S, N, B), already scaled.
    """
    FT = ftype()
    N = W.shape[0]
    tgt, src = np.nonzero(W)
    assert D[tgt, src].min() >= max(
        S, 1
    ), "a weighted delay shorter than the chunk (or zero): not this kernel"
    H = int(np.ceil((int(D.max()) + 1) / S) * S)
    wnz = jnp.asarray(W[tgt, src], FT)[:, None]
    Dnz = jnp.asarray(D[tgt, src], jnp.int32)
    srcj = jnp.asarray(src, jnp.int32)
    tgtj = jnp.asarray(tgt, jnp.int32)
    G, eta, J, Delta = (jnp.asarray(P[k], FT) for k in ("G", "eta", "J", "Delta"))
    dt_ = FT(dt)
    pi = FT(np.pi)

    def f(r, V, c):
        dr = 1 / TAU * (Delta / (pi * TAU) + 2 * V * r)
        dV = 1 / TAU * (V * V - pi * pi * TAU * TAU * r * r + eta + J * TAU * r + c)
        return dr, dV

    def gather(buf, i):
        """G * sum_j W[n, j] * r_j(i - D[n, j]) over the nonzero links, from a buffer holding columns up to i - 1."""
        rows = jnp.take(
            buf.reshape(H * N, buf.shape[-1]), ((i - Dnz) % H) * N + srcj, axis=0, mode="clip"
        )
        return G * jax.ops.segment_sum(rows * wnz, tgtj, num_segments=N, indices_are_sorted=True)

    def sample(carry, noise):
        r, V, c0, buf, i = carry

        def step(c, x):
            r, V, c0 = c
            k, nr, nV = x
            dr0, dV0 = f(r, V, c0)
            ri = jnp.maximum(r + dt_ * dr0 + nr, 0)
            Vi = V + dt_ * dV0 + nV
            c1 = gather(buf, i + k)
            dr1, dV1 = f(ri, Vi, c1)
            rn = jnp.maximum(r + dt_ * (dr0 + dr1) / 2 + nr, 0)
            Vn = V + dt_ * (dV0 + dV1) / 2 + nV
            return (rn, Vn, c1), (rn, Vn)

        (r, V, c0), (rs, Vs) = jax.lax.scan(
            step, (r, V, c0), (jnp.arange(S, dtype=jnp.int32), noise[0], noise[1])
        )
        buf = jax.lax.dynamic_update_slice(
            buf, rs, (i % H, jnp.int32(0), jnp.int32(0))
        )  # i and H are multiples of S: no wrap
        return (r, V, c0, buf, i + S), (rs, Vs)

    return sample, gather, H


def start_state(hist_r, V_last, gather, H):
    """hist_r: (Hh, N, B) columns of r, oldest first (Hh <= H); V_last (N, B). The history is placed at the end of the
    ring so that the first computed column has index H (a multiple of the chunk)."""
    FT = ftype()
    Hh = hist_r.shape[0]
    buf = jnp.zeros((H,) + hist_r.shape[1:], FT).at[H - Hh :].set(jnp.asarray(hist_r, FT))
    i0 = jnp.int32(H)
    return (buf[H - 1], jnp.asarray(V_last, FT), gather(buf, i0 - 1), buf, i0)


def build(W, L, speed, dt, params, seeds, n_tr_block=7, debug_r=False):
    """Production runner. params: (C, 5) rows (G, eta, J, Delta, sigma); seeds: list. Batch B = C * S, row b = (config b % C, seed b // C).
    Returns dict with init(), warm(carry, n_units), block(carry, first_unit) and sizes."""
    FT = ftype()
    W = np.asarray(W, np.float64)
    N = W.shape[0]
    params = np.asarray(params, np.float64)
    C, S = len(params), len(seeds)
    B = C * S
    tile = lambda k: np.tile(params[:, k], S)[None, :]
    P = dict(G=tile(0), eta=tile(1), J=tile(2), Delta=tile(3))
    sigma = jnp.asarray(tile(4), FT)
    seed_of = np.repeat(np.arange(S), C)
    steps_per_sample = int(round(TAVG_PERIOD / dt))
    assert abs(steps_per_sample * dt - TAVG_PERIOD) < 1e-12
    D = delays(L, speed, dt)
    sample, gather, H = make_sample(W, D, dt, P, steps_per_sample)
    Hh = int(D.max()) + 1
    keys = jnp.stack([jax.random.PRNGKey(int(sd)) for sd in seeds])
    sr = jnp.sqrt(2 * sigma * FT(dt))
    sV = jnp.sqrt(4 * sigma * FT(dt))
    seed_idx = jnp.asarray(seed_of)
    b = BALLOON
    k1, k2, k3 = 7 * b["E0"], 2.0, 2 * b["E0"] - 0.2

    def init():
        """As TVB's default initial history: r uniform on [0, 2], V uniform on [-2, 1.5], independent per node and time."""
        r_hist = np.stack(
            [np.random.default_rng(1000 * int(sd) + 11).uniform(0.0, 2.0, (Hh, N)) for sd in seeds]
        )[seed_of]
        V0 = np.stack(
            [np.random.default_rng(1000 * int(sd) + 12).uniform(-2.0, 1.5, N) for sd in seeds]
        )[seed_of]
        return start_state(np.transpose(r_hist, (1, 2, 0)), V0.T, gather, H)

    def one_sample(state, sample_index):
        z = jax.vmap(
            lambda k: jax.random.normal(
                jax.random.fold_in(k, sample_index), (steps_per_sample, 2, N), FT
            )
        )(
            keys
        )  # (S, steps, 2, N)
        z = jnp.transpose(z[seed_idx], (1, 2, 3, 0))  # (steps, 2, N, B)
        state, (rs, _) = sample(state, (sr * z[:, 0], sV * z[:, 1]))
        return state, rs.mean(0)  # temporal average over the sample, (N, B)

    def hemo_step(h, u):
        s_, f_, v_, q_ = h
        ds = u - b["kappa"] * s_ - b["gamma"] * (f_ - 1)
        dv = (f_ - v_ ** (1 / b["alpha"])) / b["tau"]
        dq = (f_ * (1 - (1 - b["E0"]) ** (1 / f_)) / b["E0"] - q_ * v_ ** (1 / b["alpha"] - 1)) / b[
            "tau"
        ]
        s_ = s_ + 1e-3 * ds
        f_ = jnp.maximum(f_ + 1e-3 * s_, 1e-3)  # as mpr_net.Balloon: its df aliases the updated s
        return (s_, f_, jnp.maximum(v_ + 1e-3 * dv, 1e-3), jnp.maximum(q_ + 1e-3 * dq, 1e-3))

    def one_unit(carry, unit_index):
        """Ten tavg samples (one time unit). Statistics on the tavg rate, as tvb_ref_run.statistics."""
        state, hemo, acc = carry

        def body(c, k):
            state, hemo, (sum_r, up7, up5, mx) = c
            state, rbar = one_sample(state, unit_index * SAMPLES_PER_UNIT + k)
            hemo = hemo_step(hemo, rbar)
            acc_ = (
                sum_r + rbar,
                up7 + (rbar > 0.7),
                up5 + (rbar > 0.5),
                jnp.maximum(mx, rbar.max(0)),
            )
            return (state, hemo, acc_), rbar

        (state, hemo, acc), rbars = jax.lax.scan(
            body, (state, hemo, acc), jnp.arange(SAMPLES_PER_UNIT)
        )
        return (state, hemo, acc), (
            rbars if debug_r else rbars[-1]
        )  # rbars[-1] is the sample r[9::10] uses

    def fresh_hemo():
        o = jnp.ones((N, B), FT)
        return (jnp.zeros((N, B), FT), o, o, o)

    def fresh_acc():
        z = jnp.zeros((N, B), FT)
        return (z, z, z, jnp.zeros((B,), FT))

    @jax.jit
    def warm(state, first_unit, n_units_arr):
        """Advance without keeping statistics; n_units_arr = arange(n_units) (its length fixes the compiled size)."""
        (state, _, _), _ = jax.lax.scan(
            one_unit, (state, fresh_hemo(), fresh_acc()), first_unit + n_units_arr
        )
        return state

    @jax.jit
    def block(state, hemo, first_unit):
        """n_tr_block TRs. Per TR: BOLD at the TR's end, per-node sums of r and of the up indicators (each (N, B)), max r (B,).
        Over the block: sum of x (N, B) and of x x^T (B, N, N) for x = the last tavg sample of each time unit.
        """

        def one_tr(c, tr):
            state, hemo, sx, sxx = c
            (state, hemo, acc), xs = jax.lax.scan(
                one_unit,
                (state, hemo, fresh_acc()),
                first_unit + tr * UNITS_PER_TR + jnp.arange(UNITS_PER_TR),
            )
            s_, f_, v_, q_ = hemo
            bold = b["V0"] * (k1 * (1 - q_) + k2 * (1 - q_ / v_) + k3 * (1 - v_))
            xs_all = xs
            xs = (
                xs[:, -1] if debug_r else xs
            )  # debug_r: also return every tavg sample, (72, 10, N, B)
            sx = sx + xs.sum(0)
            sxx = sxx + jnp.einsum("tib,tjb->bij", xs, xs)
            return (state, hemo, sx, sxx), (bold, acc[0], acc[1], acc[2], acc[3]) + (
                (xs_all,) if debug_r else ()
            )

        c0 = (state, hemo, jnp.zeros((N, B), FT), jnp.zeros((B, N, N), FT))
        (state, hemo, sx, sxx), per_tr = jax.lax.scan(one_tr, c0, jnp.arange(n_tr_block))
        finite = jnp.isfinite(state[0]).all(0) & jnp.isfinite(state[1]).all(0)
        return state, hemo, per_tr, sx, sxx, finite

    return dict(
        init=init,
        warm=warm,
        block=block,
        fresh_hemo=fresh_hemo,
        B=B,
        C=C,
        S=S,
        N=N,
        H=H,
        steps_per_sample=steps_per_sample,
        n_tr_block=n_tr_block,
        seed_of=seed_of,
        config_of=np.tile(np.arange(C), S),
        D=D,
    )
