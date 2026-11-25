"""The simulation: a network of Montbrió–Pazó–Roxin (MPR) regions with delayed coupling, written in JAX.

It reproduces the kernel The Virtual Brain runs for this model (tvb-library 2.10.0, numba backend, template
nb-montbrio.py.mako, default mode), for networks in which every connection has a delay of at least one step.
`src/validation/mprjax_match.py` checks the match step by step against TVB's own buffers.

The model
---------
Each region has a firing rate r and a mean membrane potential V:

    dr/dt = Delta / pi + 2 r V
    dV/dt = V^2 - pi^2 r^2 + eta + J r + G * (input from the other regions)

The input to region n is  sum_j W[n, j] * r_j(t - delay[n, j]):  the past rates of the regions it is connected to,
weighted by the connectome W.

One integration step (stochastic Heun, as in TVB)
-------------------------------------------------
    coupling_before = G * sum_j W[n, j] * r_j(step - 1 - delay)
    predictor       = state + dt * f(state, coupling_before) + noise            (r kept >= 0)
    coupling_after  = G * sum_j W[n, j] * r_j(step - delay)
    new state       = state + dt * (f(state, coupling_before) + f(predictor, coupling_after)) / 2 + the same noise
                                                                                (r kept >= 0)

Because every delay is at least one step, the "after" coupling of one step is the "before" coupling of the next, so
the coupling is computed once per step and carried over. (With zero delays TVB's kernel reads the column it is
writing; that case is NOT handled here. See doc/LOG.md, 2026-11-24, 10:20.)

Noise, as TVB's additive noise with nsig = [sigma, 2 sigma]: each step adds sqrt(2 sigma dt) z to r and
sqrt(4 sigma dt) z to V, with z standard normal.

Batching
--------
Many simulations run at once. The last axis of every array is the batch: one column per (parameter setting, seed).
Each column carries its own G, eta, J, Delta and sigma. Columns that share a seed share the noise and the starting
state.
"""
import jax
import jax.numpy as jnp
import numpy as np

TAU = 1.0
TAVG_PERIOD = 0.1  # the output sample: TVB's TemporalAverage period, in time units
SAMPLES_PER_UNIT = 10  # output samples per time unit
UNITS_PER_TR = 72  # one scan volume = 720 output samples (the group reads one sample as 1 ms)
BALLOON = dict(kappa=0.65, gamma=0.41, tau=0.98, alpha=0.32, E0=0.4, V0=0.04)  # as mpr_net.Balloon


def ftype():
    """Single precision unless JAX was started with JAX_ENABLE_X64=1."""
    return np.float64 if jax.config.jax_enable_x64 else np.float32


def delays(tract_lengths, speed, dt):
    """Delay of every connection in steps, as TVB computes it: rint(tract_length / speed / dt)."""
    return np.rint(tract_lengths / speed / dt).astype(np.int32)


def make_sample(weights, delay_steps, dt, parameters, steps_per_sample):
    """Build the function that advances the network by one output sample (`steps_per_sample` steps).

    weights          (regions, regions) connectome
    delay_steps      (regions, regions) delay of each connection, in steps
    parameters       dict of G, eta, J, Delta, each of shape (1, batch)
    steps_per_sample steps in one output sample

    Returns (sample, gather, history_length).

    sample(carry, noise) -> (carry, (r, V))
        carry = (r, V, coupling, history, step): r, V and coupling are (regions, batch); history is the ring buffer
        of past rates, (history_length, regions, batch), with step t stored in slot t mod history_length.
        noise = (noise_r, noise_V), each (steps_per_sample, regions, batch), already scaled.
        The returned r and V hold every step of the sample: (steps_per_sample, regions, batch).

    Why a whole output sample at a time: every connection's delay is at least one output sample long, so all the
    past rates needed inside the sample were written before it began. The history can therefore stay fixed during
    the sample and be updated once at its end. Updating it every step made JAX copy the whole buffer each step and
    was 20 times slower on the GPU, for the same numbers.
    """
    FT = ftype()
    n_regions = weights.shape[0]

    # only the connections that exist
    target, source = np.nonzero(weights)
    assert delay_steps[target, source].min() >= max(
        steps_per_sample, 1
    ), "a connection's delay is shorter than one output sample (or zero): this kernel does not handle that"
    history_length = int(
        np.ceil((int(delay_steps.max()) + 1) / steps_per_sample) * steps_per_sample
    )
    link_weight = jnp.asarray(weights[target, source], FT)[:, None]
    link_delay = jnp.asarray(delay_steps[target, source], jnp.int32)
    link_source = jnp.asarray(source, jnp.int32)
    link_target = jnp.asarray(target, jnp.int32)

    G, eta, J, Delta = (jnp.asarray(parameters[name], FT) for name in ("G", "eta", "J", "Delta"))
    dt_ = FT(dt)
    pi = FT(np.pi)

    def f(r, V, coupling):
        """Right-hand side of the model: (dr/dt, dV/dt)."""
        dr = 1 / TAU * (Delta / (pi * TAU) + 2 * V * r)
        dV = 1 / TAU * (V * V - pi * pi * TAU * TAU * r * r + eta + J * TAU * r + coupling)
        return dr, dV

    def gather(history, step):
        """Coupling into every region at `step`: G * sum_j W[n, j] * r_j(step - delay[n, j])."""
        flat_history = history.reshape(history_length * n_regions, history.shape[-1])
        slot = ((step - link_delay) % history_length) * n_regions + link_source
        delayed_rates = jnp.take(flat_history, slot, axis=0, mode="clip")
        return G * jax.ops.segment_sum(
            delayed_rates * link_weight,
            link_target,
            num_segments=n_regions,
            indices_are_sorted=True,
        )

    def sample(carry, noise):
        r, V, coupling_before, history, first_step = carry

        def heun_step(state, inputs):
            r, V, coupling_before = state
            offset, noise_r, noise_V = inputs
            dr0, dV0 = f(r, V, coupling_before)
            r_predicted = jnp.maximum(r + dt_ * dr0 + noise_r, 0)
            V_predicted = V + dt_ * dV0 + noise_V
            coupling_after = gather(history, first_step + offset)
            dr1, dV1 = f(r_predicted, V_predicted, coupling_after)
            r_new = jnp.maximum(r + dt_ * (dr0 + dr1) / 2 + noise_r, 0)
            V_new = V + dt_ * (dV0 + dV1) / 2 + noise_V
            return (r_new, V_new, coupling_after), (r_new, V_new)

        (r, V, coupling_before), (r_steps, V_steps) = jax.lax.scan(
            heun_step,
            (r, V, coupling_before),
            (jnp.arange(steps_per_sample, dtype=jnp.int32), noise[0], noise[1]),
        )
        # write the sample's rates into the history in one piece (it never wraps: both lengths are multiples
        # of steps_per_sample)
        history = jax.lax.dynamic_update_slice(
            history, r_steps, (first_step % history_length, jnp.int32(0), jnp.int32(0))
        )
        return (r, V, coupling_before, history, first_step + steps_per_sample), (r_steps, V_steps)

    return sample, gather, history_length


def start_state(rate_history, V_last, gather, history_length):
    """The carry to start from.

    rate_history  (steps, regions, batch) past rates, oldest first; at most history_length steps
    V_last        (regions, batch) potentials at the last of those steps

    The history is placed at the end of the ring, so the first computed step has index history_length.
    """
    FT = ftype()
    n_given = rate_history.shape[0]
    history = (
        jnp.zeros((history_length,) + rate_history.shape[1:], FT)
        .at[history_length - n_given :]
        .set(jnp.asarray(rate_history, FT))
    )
    first_step = jnp.int32(history_length)
    return (
        history[history_length - 1],
        jnp.asarray(V_last, FT),
        gather(history, first_step - 1),
        history,
        first_step,
    )


def build(W, L, speed, dt, params, seeds, n_tr_block=7, debug_r=False):
    """Set up a batch of full runs, with the statistics accumulated on the device as the runs go.

    W, L     connectome weights and tract lengths
    speed    conduction speed (delay = tract length / speed)
    params   (settings, 5) rows of (G, eta, J, Delta, sigma)
    seeds    list of seeds; every setting is run with every seed

    The batch has settings x seeds columns; column b is setting b % settings with seed b // settings.

    Returns a dict:
        init()                          -> starting state
        warm(state, first_unit, units)  -> state after advancing without keeping statistics
        block(state, hemo, first_unit)  -> state, hemo, per-volume numbers, sums for the covariance, finite flags
        fresh_hemo()                    -> resting state of the BOLD model
        and the sizes (B, C, S, N, H, steps_per_sample, n_tr_block) and index arrays (seed_of, config_of, D).
    """
    FT = ftype()
    W = np.asarray(W, np.float64)
    n_regions = W.shape[0]
    params = np.asarray(params, np.float64)
    n_settings, n_seeds = len(params), len(seeds)
    batch = n_settings * n_seeds

    # one column per (setting, seed)
    column = lambda k: np.tile(params[:, k], n_seeds)[None, :]
    parameters = dict(G=column(0), eta=column(1), J=column(2), Delta=column(3))
    sigma = jnp.asarray(column(4), FT)
    seed_of = np.repeat(np.arange(n_seeds), n_settings)
    seed_index = jnp.asarray(seed_of)

    steps_per_sample = int(round(TAVG_PERIOD / dt))
    assert abs(steps_per_sample * dt - TAVG_PERIOD) < 1e-12
    delay_steps = delays(L, speed, dt)
    sample, gather, history_length = make_sample(W, delay_steps, dt, parameters, steps_per_sample)
    n_history_steps = int(delay_steps.max()) + 1

    keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
    noise_scale_r = jnp.sqrt(2 * sigma * FT(dt))
    noise_scale_V = jnp.sqrt(4 * sigma * FT(dt))

    b = BALLOON
    k1, k2, k3 = 7 * b["E0"], 2.0, 2 * b["E0"] - 0.2

    def init():
        """Starting state as TVB's default: past rates uniform on [0, 2], potentials uniform on [-2, 1.5],
        independent for every region and step. One draw per seed."""
        rate_history = np.stack(
            [
                np.random.default_rng(1000 * int(seed) + 11).uniform(
                    0.0, 2.0, (n_history_steps, n_regions)
                )
                for seed in seeds
            ]
        )[seed_of]
        V_start = np.stack(
            [
                np.random.default_rng(1000 * int(seed) + 12).uniform(-2.0, 1.5, n_regions)
                for seed in seeds
            ]
        )[seed_of]
        return start_state(np.transpose(rate_history, (1, 2, 0)), V_start.T, gather, history_length)

    def one_sample(state, sample_index):
        """Advance one output sample and return its time-averaged rate, (regions, batch)."""
        z = jax.vmap(
            lambda key: jax.random.normal(
                jax.random.fold_in(key, sample_index), (steps_per_sample, 2, n_regions), FT
            )
        )(
            keys
        )  # (seeds, steps, 2, regions)
        z = jnp.transpose(z[seed_index], (1, 2, 3, 0))  # (steps, 2, regions, batch)
        state, (r_steps, _) = sample(state, (noise_scale_r * z[:, 0], noise_scale_V * z[:, 1]))
        return state, r_steps.mean(0)

    def hemo_step(hemo, rate):
        """One step of the Balloon–Windkessel BOLD model, driven by the rate (step 1e-3, as mpr_net.Balloon)."""
        s_, f_, v_, q_ = hemo
        ds = rate - b["kappa"] * s_ - b["gamma"] * (f_ - 1)
        dv = (f_ - v_ ** (1 / b["alpha"])) / b["tau"]
        dq = (f_ * (1 - (1 - b["E0"]) ** (1 / f_)) / b["E0"] - q_ * v_ ** (1 / b["alpha"] - 1)) / b[
            "tau"
        ]
        s_ = s_ + 1e-3 * ds
        f_ = jnp.maximum(
            f_ + 1e-3 * s_, 1e-3
        )  # as mpr_net.Balloon: its flow update uses the updated s
        return (s_, f_, jnp.maximum(v_ + 1e-3 * dv, 1e-3), jnp.maximum(q_ + 1e-3 * dq, 1e-3))

    def one_unit(carry, unit_index):
        """Advance one time unit (ten output samples), updating the BOLD model and the running sums."""
        state, hemo, sums = carry

        def one_output_sample(inner, k):
            state, hemo, (sum_rate, count_above_07, count_above_05, largest_rate) = inner
            state, rate = one_sample(state, unit_index * SAMPLES_PER_UNIT + k)
            hemo = hemo_step(hemo, rate)
            sums = (
                sum_rate + rate,
                count_above_07 + (rate > 0.7),
                count_above_05 + (rate > 0.5),
                jnp.maximum(largest_rate, rate.max(0)),
            )
            return (state, hemo, sums), rate

        (state, hemo, sums), rates = jax.lax.scan(
            one_output_sample, (state, hemo, sums), jnp.arange(SAMPLES_PER_UNIT)
        )
        # the last sample of the unit is the one the covariance statistics use; debug_r returns all ten
        return (state, hemo, sums), (rates if debug_r else rates[-1])

    def fresh_hemo():
        ones = jnp.ones((n_regions, batch), FT)
        return (jnp.zeros((n_regions, batch), FT), ones, ones, ones)

    def fresh_sums():
        zeros = jnp.zeros((n_regions, batch), FT)
        return (zeros, zeros, zeros, jnp.zeros((batch,), FT))

    @jax.jit
    def warm(state, first_unit, n_units_arr):
        """Advance without keeping statistics. n_units_arr = arange(number of time units)."""
        (state, _, _), _ = jax.lax.scan(
            one_unit, (state, fresh_hemo(), fresh_sums()), first_unit + n_units_arr
        )
        return state

    @jax.jit
    def block(state, hemo, first_unit):
        """Advance n_tr_block scan volumes.

        Per volume: BOLD at its end; per-region sums of the rate and of the two "up" indicators, each
        (regions, batch); the largest rate, (batch,).
        Over the block: sum of x, (regions, batch), and of x x^T, (batch, regions, regions), where x is the last
        output sample of each time unit. These give the covariance of the rates.
        """

        def one_volume(carry, volume):
            state, hemo, sum_x, sum_xx = carry
            (state, hemo, sums), x = jax.lax.scan(
                one_unit,
                (state, hemo, fresh_sums()),
                first_unit + volume * UNITS_PER_TR + jnp.arange(UNITS_PER_TR),
            )
            s_, f_, v_, q_ = hemo
            bold = b["V0"] * (k1 * (1 - q_) + k2 * (1 - q_ / v_) + k3 * (1 - v_))
            every_sample = x
            x = (
                x[:, -1] if debug_r else x
            )  # debug_r: x holds all ten samples, (72, 10, regions, batch)
            sum_x = sum_x + x.sum(0)
            sum_xx = sum_xx + jnp.einsum("tib,tjb->bij", x, x)
            return (state, hemo, sum_x, sum_xx), (bold, sums[0], sums[1], sums[2], sums[3]) + (
                (every_sample,) if debug_r else ()
            )

        start = (
            state,
            hemo,
            jnp.zeros((n_regions, batch), FT),
            jnp.zeros((batch, n_regions, n_regions), FT),
        )
        (state, hemo, sum_x, sum_xx), per_volume = jax.lax.scan(
            one_volume, start, jnp.arange(n_tr_block)
        )
        finite = jnp.isfinite(state[0]).all(0) & jnp.isfinite(state[1]).all(0)
        return state, hemo, per_volume, sum_x, sum_xx, finite

    return dict(
        init=init,
        warm=warm,
        block=block,
        fresh_hemo=fresh_hemo,
        B=batch,
        C=n_settings,
        S=n_seeds,
        N=n_regions,
        H=history_length,
        steps_per_sample=steps_per_sample,
        n_tr_block=n_tr_block,
        seed_of=seed_of,
        config_of=np.tile(np.arange(n_settings), n_seeds),
        D=delay_steps,
    )
