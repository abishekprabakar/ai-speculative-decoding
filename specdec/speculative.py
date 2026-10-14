"""Speculative sampling (Leviathan et al. 2023; Chen et al. 2023).

One iteration, with draft q, target p and lookahead gamma:

1. The draft proposes x_1..x_gamma autoregressively, x_i ~ q(. | prefix, x_<i).
2. The target scores all gamma+1 positions in ONE batched forward pass.
3. For i = 1..gamma: accept x_i with probability min(1, p(x_i)/q(x_i)).
   On the first rejection, emit a token from the residual
       r(x) = max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y))
   and stop.
4. If all gamma drafts are accepted, emit a bonus token from p at position gamma+1.

Each emitted token is distributed exactly as the target would have produced it:
    P(emit x) = q(x) min(1, p(x)/q(x)) + (1 - alpha) r(x)
              = min(q(x), p(x)) + max(0, p(x) - q(x)) = p(x),
with alpha = sum_x min(p(x), q(x)). ``theory.exact_sequence_distribution``
enumerates the whole process to check this to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a distribution by temperature; T=0 is greedy (one-hot argmax)."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if temperature == 0:
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out
    if temperature == 1:
        return p
    logp = np.log(np.clip(p, 1e-300, None)) / temperature
    logp -= logp.max()
    w = np.exp(logp) * (p > 0)
    return w / w.sum()


def residual(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalised max(0, p - q). Falls back to p if p == q to rounding."""
    r = np.maximum(p - q, 0.0)
    s = r.sum()
    if s <= 1e-15:
        return p
    return r / s


def _sample(dist: np.ndarray, rng: np.random.Generator) -> int:
    # inverse-CDF on a single uniform; robust to tiny normalisation error
    c = np.cumsum(dist)
    return int(min(np.searchsorted(c, rng.random() * c[-1], side="right"), len(dist) - 1))


@dataclass
class DecodeStats:
    iterations: int = 0          # speculative iterations (== target forward passes)
    drafted: int = 0             # draft tokens proposed
    accepted: int = 0            # draft tokens accepted
    emitted: int = 0             # tokens appended to the output
    alpha_sum: float = 0.0       # sum over drafted positions of sum_x min(p, q)
    chunk_lengths: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else float("nan")

    @property
    def tokens_per_iteration(self) -> float:
        return self.emitted / self.iterations if self.iterations else float("nan")

    @property
    def mean_alpha(self) -> float:
        """Average of the per-position acceptance probability alpha = 1 - TV(p, q)."""
        return self.alpha_sum / self.drafted if self.drafted else float("nan")


def speculative_step(target, draft, prefix, gamma: int, rng: np.random.Generator,
                     temperature: float = 1.0, stats: DecodeStats | None = None) -> list[int]:
    """Run one speculative iteration and return the 1..gamma+1 emitted tokens."""
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    ctx = list(prefix)
    drafts: list[int] = []
    qs: list[np.ndarray] = []
    for _ in range(gamma):
        q = apply_temperature(draft.dist(ctx), temperature)
        x = _sample(q, rng)
        drafts.append(x)
        qs.append(q)
        ctx.append(x)

    ps = [apply_temperature(p, temperature) for p in target.dist_batch(prefix, drafts)]

    out: list[int] = []
    for i, x in enumerate(drafts):
        p, q = ps[i], qs[i]
        if stats is not None:
            stats.drafted += 1
            stats.alpha_sum += float(np.minimum(p, q).sum())
        # q[x] > 0 because x was sampled from q
        if rng.random() * q[x] < p[x]:          # i.e. u < min(1, p/q) without dividing
            out.append(x)
            if stats is not None:
                stats.accepted += 1
        else:
            out.append(_sample(residual(p, q), rng))
            break
    else:
        out.append(_sample(ps[gamma], rng))

    if stats is not None:
        stats.iterations += 1
        stats.emitted += len(out)
        stats.chunk_lengths.append(len(out))
    return out


def speculative_generate(target, draft, prefix, n_tokens: int, gamma: int,
                         rng: np.random.Generator, temperature: float = 1.0,
                         stop_token: int | None = None) -> tuple[list[int], DecodeStats]:
    """Generate ``n_tokens`` new tokens (truncating the last chunk)."""
    stats = DecodeStats()
    seq = list(prefix)
    produced: list[int] = []
    while len(produced) < n_tokens:
        chunk = speculative_step(target, draft, seq, gamma, rng, temperature, stats)
        for x in chunk:
            if len(produced) == n_tokens:
                break
            produced.append(x)
            seq.append(x)
            if stop_token is not None and x == stop_token:
                return produced, stats
    return produced, stats


def autoregressive_generate(model, prefix, n_tokens: int, rng: np.random.Generator,
                            temperature: float = 1.0, stop_token: int | None = None) -> list[int]:
    """Plain one-token-per-forward-pass sampling from ``model``."""
    seq = list(prefix)
    out: list[int] = []
    for _ in range(n_tokens):
        x = _sample(apply_temperature(model.dist(seq), temperature), rng)
        out.append(x)
        seq.append(x)
        if stop_token is not None and x == stop_token:
            break
    return out
