"""Closed-form analysis of speculative decoding, and an exact enumerator.

Notation follows Leviathan et al. (2023), "Fast Inference from Transformers via
Speculative Decoding":

    alpha      = E[ sum_x min(p(x), q(x)) ] = 1 - TV(p, q)   (acceptance prob.)
    gamma      = number of draft tokens per iteration
    c          = cost of one draft forward pass / cost of one target forward pass

If alpha is the same at every position (i.i.d. acceptances), the number of
tokens emitted by one iteration is a truncated geometric variable, and

    E[tokens / iteration]  = (1 - alpha^(gamma+1)) / (1 - alpha)
    walltime improvement   = (1 - alpha^(gamma+1)) / ((1 - alpha) (gamma c + 1))
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .speculative import apply_temperature, residual


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------

def acceptance_probability(p: np.ndarray, q: np.ndarray) -> float:
    """alpha = sum_x min(p(x), q(x)) -- the chance a single draft token survives."""
    return float(np.minimum(p, q).sum())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def expected_tokens_per_iteration(alpha: float, gamma: int) -> float:
    """E[#emitted] = 1 + alpha + ... + alpha^gamma."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def chunk_length_pmf(alpha: float, gamma: int) -> np.ndarray:
    """P(iteration emits n tokens), n = 1..gamma+1 (index 0 unused, = 0)."""
    pmf = np.zeros(gamma + 2)
    for n in range(1, gamma + 1):
        pmf[n] = alpha ** (n - 1) * (1 - alpha)
    pmf[gamma + 1] = alpha ** gamma
    return pmf


def walltime_improvement(alpha: float, gamma: int, c: float) -> float:
    """Expected speedup over plain target sampling, in units of target passes."""
    return expected_tokens_per_iteration(alpha, gamma) / (gamma * c + 1.0)


def optimal_gamma(alpha: float, c: float, max_gamma: int = 64) -> int:
    """gamma in [0, max_gamma] maximising walltime_improvement."""
    return int(max(range(max_gamma + 1), key=lambda g: walltime_improvement(alpha, g, c)))


# --------------------------------------------------------------------------
# Exact enumeration of the speculative process (small vocabularies only)
# --------------------------------------------------------------------------

def exact_step_distribution(target_fn, draft_fn, context, gamma: int,
                            temperature: float = 1.0) -> dict[tuple, float]:
    """Exact distribution over the token chunk one speculative iteration emits.

    ``target_fn`` and ``draft_fn`` map a context tuple to a probability vector.
    Enumerates every draft path, every accept/reject outcome and every residual
    sample, so the returned probabilities are exact (up to float rounding).
    """
    out: dict[tuple, float] = defaultdict(float)

    def rec(ctx: tuple, i: int, mass: float, emitted: tuple) -> None:
        p = apply_temperature(np.asarray(target_fn(ctx), float), temperature)
        if i == gamma:                                  # bonus token from p
            for y, py in enumerate(p):
                if py > 0:
                    out[emitted + (y,)] += mass * py
            return
        q = apply_temperature(np.asarray(draft_fn(ctx), float), temperature)
        reject_mass = 0.0
        for x, qx in enumerate(q):
            if qx <= 0:
                continue
            acc = min(1.0, p[x] / qx)
            reject_mass += qx * (1.0 - acc)
            if acc > 0:
                rec(ctx + (x,), i + 1, mass * qx * acc, emitted + (x,))
        if reject_mass > 0:
            r = residual(p, q)
            for y, ry in enumerate(r):
                if ry > 0:
                    out[emitted + (y,)] += mass * reject_mass * ry

    rec(tuple(context), 0, 1.0, ())
    return dict(out)


def exact_sequence_distribution(target_fn, draft_fn, prefix, length: int, gamma: int,
                                temperature: float = 1.0) -> dict[tuple, float]:
    """Exact law of the first ``length`` tokens produced by speculative decoding."""
    done: dict[tuple, float] = defaultdict(float)
    frontier: dict[tuple, float] = {(): 1.0}
    prefix = tuple(prefix)
    while frontier:
        nxt: dict[tuple, float] = defaultdict(float)
        for produced, mass in frontier.items():
            for chunk, pc in exact_step_distribution(target_fn, draft_fn, prefix + produced,
                                                     gamma, temperature).items():
                seq = (produced + chunk)[:length]
                if len(seq) == length:
                    done[seq] += mass * pc
                else:
                    nxt[seq] += mass * pc
        frontier = nxt
    return dict(done)


def target_sequence_distribution(target_fn, prefix, length: int,
                                 temperature: float = 1.0) -> dict[tuple, float]:
    """Exact law of ``length`` tokens sampled autoregressively from the target."""
    dist: dict[tuple, float] = {(): 1.0}
    prefix = tuple(prefix)
    for _ in range(length):
        nxt: dict[tuple, float] = {}
        for seq, m in dist.items():
            p = apply_temperature(np.asarray(target_fn(prefix + seq), float), temperature)
            for y, py in enumerate(p):
                if py > 0:
                    nxt[seq + (y,)] = m * py
        dist = nxt
    return dist


def expected_chunk_length_exact(target_fn, draft_fn, context, gamma: int,
                                temperature: float = 1.0) -> float:
    return sum(len(k) * v for k, v in
               exact_step_distribution(target_fn, draft_fn, context, gamma, temperature).items())
