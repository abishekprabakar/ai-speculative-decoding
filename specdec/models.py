"""Character-level n-gram language models and a procedural training corpus.

The point of these models is not quality but *structure*: speculative decoding
needs a cheap draft model q and an expensive target model p over the same
vocabulary, and the acceptance rate depends only on how close q is to p in each
context. Interpolated n-gram models of different orders give exactly that: a
high-order target and a low-order draft that agree on easy characters (after
"th" comes "e") and disagree on hard ones (the first letter of a word).

Every model exposes ``dist(context) -> np.ndarray`` (a probability vector over
the vocabulary) and counts its forward passes, so the decoding loop can report
target-model calls honestly.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------
# Procedural corpus: a tiny stochastic grammar, so the repo needs no download
# and every run (and test) sees the same text for a given seed.
# --------------------------------------------------------------------------

_DET = ["the", "a", "every", "some", "this", "that"]
_ADJ = ["quick", "lazy", "small", "bright", "quiet", "old", "red", "clever", "tired", "brave"]
_NOUN = ["cat", "dog", "fox", "bird", "child", "robot", "farmer", "teacher", "river", "garden",
         "machine", "student", "painter", "sailor"]
_VERB_T = ["sees", "likes", "follows", "finds", "watches", "helps", "chases", "draws", "builds"]
_VERB_I = ["sleeps", "runs", "waits", "sings", "laughs", "thinks", "rests", "wanders"]
_PREP = ["near", "under", "behind", "beside", "over", "with"]
_ADV = ["slowly", "quietly", "often", "never", "always", "happily"]


def _np(rng: np.random.Generator) -> str:
    words = [rng.choice(_DET)]
    if rng.random() < 0.5:
        words.append(rng.choice(_ADJ))
    words.append(rng.choice(_NOUN))
    return " ".join(words)


def _sentence(rng: np.random.Generator) -> str:
    subj = _np(rng)
    if rng.random() < 0.55:
        vp = f"{rng.choice(_VERB_T)} {_np(rng)}"
    else:
        vp = str(rng.choice(_VERB_I))
    if rng.random() < 0.3:
        vp = f"{rng.choice(_ADV)} {vp}"
    if rng.random() < 0.35:
        vp = f"{vp} {rng.choice(_PREP)} {_np(rng)}"
    return f"{subj} {vp}."


def generate_corpus(n_sentences: int, seed: int = 0) -> str:
    """Return ``n_sentences`` grammar sentences joined by single spaces."""
    rng = np.random.default_rng(seed)
    return " ".join(_sentence(rng) for _ in range(n_sentences))


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

class Vocab:
    def __init__(self, chars: str):
        self.itos = sorted(set(chars))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)


# --------------------------------------------------------------------------
# Interpolated n-gram model
# --------------------------------------------------------------------------

class NGramLM:
    """Jelinek-Mercer interpolated character n-gram model.

    P(x | h) = sum_k lambda_k * P_k(x | last k chars of h), k = 0..order,
    where P_0 is add-one unigram and each P_k (k>=1) is the ML estimate; when a
    history of length k was never seen its weight is folded into lower orders.
    The result is a proper distribution in every context, with full support
    (the unigram is add-one), which keeps draft probabilities strictly positive.
    """

    def __init__(self, vocab: Vocab, order: int, lambdas=None):
        self.vocab = vocab
        self.order = order
        V = len(vocab)
        if lambdas is None:
            # More weight on longer contexts, but never zero on any order.
            raw = np.array([2.0 ** k for k in range(order + 1)])
            lambdas = raw / raw.sum()
        self.lambdas = np.asarray(lambdas, dtype=float)
        if self.lambdas.shape != (order + 1,) or not math.isclose(self.lambdas.sum(), 1.0):
            raise ValueError("need order+1 interpolation weights summing to 1")
        self.counts = [defaultdict(lambda: np.zeros(V)) for _ in range(order + 1)]
        self.calls = 0  # forward passes ("one call" = one context's distribution, or one batched verify)
        self._cache: dict[tuple, np.ndarray] = {}

    def fit(self, ids: list[int]) -> "NGramLM":
        for t, x in enumerate(ids):
            for k in range(self.order + 1):
                if t - k < 0:
                    break
                self.counts[k][tuple(ids[t - k:t])][x] += 1.0
        self._cache.clear()
        return self

    def _dist_uncounted(self, context) -> np.ndarray:
        key = tuple(context[-self.order:]) if self.order else ()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        V = len(self.vocab)
        uni = self.counts[0][()] + 1.0
        out = self.lambdas[0] * uni / uni.sum()
        spare = 0.0
        for k in range(1, self.order + 1):
            if len(context) < k:
                spare += self.lambdas[k]
                continue
            h = tuple(context[-k:])
            c = self.counts[k].get(h)
            if c is None or c.sum() == 0:
                spare += self.lambdas[k]
                continue
            out = out + self.lambdas[k] * c / c.sum()
        if spare:
            # redistribute unused weight proportionally over what we have
            out = out / (1.0 - spare)
        out = out / out.sum()
        self._cache[key] = out
        assert out.shape == (V,)
        return out

    def dist(self, context) -> np.ndarray:
        """Next-token distribution after ``context``; counts one forward pass."""
        self.calls += 1
        return self._dist_uncounted(context)

    def dist_batch(self, prefix, continuation) -> list[np.ndarray]:
        """Distributions at every position of ``continuation`` plus one more.

        Returns len(continuation)+1 vectors: p(.|prefix), p(.|prefix+c[:1]), ...
        This models a transformer scoring a block of tokens in ONE forward pass,
        so it counts as a single call -- the whole reason speculative decoding
        saves time.
        """
        self.calls += 1
        ctx = list(prefix)
        out = [self._dist_uncounted(ctx)]
        for x in continuation:
            ctx.append(x)
            out.append(self._dist_uncounted(ctx))
        return out

    def log_likelihood(self, ids: list[int]) -> float:
        ll = 0.0
        for t in range(len(ids)):
            ll += math.log(self._dist_uncounted(ids[:t])[ids[t]])
        return ll

    def perplexity(self, ids: list[int]) -> float:
        return math.exp(-self.log_likelihood(ids) / len(ids))


class TableLM:
    """A model defined by an explicit function context -> distribution.

    Used by the tests to plug in hand-written Markov chains and i.i.d. models
    whose exact behaviour can be enumerated.
    """

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def dist(self, context) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.fn(tuple(context)), dtype=float)

    def dist_batch(self, prefix, continuation) -> list[np.ndarray]:
        self.calls += 1
        ctx = list(prefix)
        out = [np.asarray(self.fn(tuple(ctx)), dtype=float)]
        for x in continuation:
            ctx.append(x)
            out.append(np.asarray(self.fn(tuple(ctx)), dtype=float))
        return out
