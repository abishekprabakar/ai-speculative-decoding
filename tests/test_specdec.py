"""Tests for speculative decoding.

Expected values come from three independent sources, never from the
implementation's own output:
  * hand arithmetic on a 3-token example (worked in the comments),
  * exact enumeration of the target model's own autoregressive law,
  * the closed-form geometric results of Leviathan et al. (2023).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from specdec import (NGramLM, TableLM, Vocab, acceptance_probability, apply_temperature,  # noqa: E402
                     autoregressive_generate, chunk_length_pmf, exact_sequence_distribution,
                     exact_step_distribution, expected_tokens_per_iteration, generate_corpus,
                     optimal_gamma, residual, speculative_generate, speculative_step,
                     target_sequence_distribution, total_variation, walltime_improvement)
from specdec.speculative import DecodeStats  # noqa: E402

P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.2, 0.2, 0.6])


# ---------------------------------------------------------------- hand-worked

def test_alpha_and_tv_by_hand():
    # min(p,q) = [0.2, 0.2, 0.2] -> alpha = 0.6 ; |p-q| = [0.3,0.1,0.4] -> TV = 0.4
    assert acceptance_probability(P, Q) == pytest.approx(0.6)
    assert total_variation(P, Q) == pytest.approx(0.4)
    assert acceptance_probability(P, Q) == pytest.approx(1 - total_variation(P, Q))


def test_residual_by_hand():
    # max(0, p-q) = [0.3, 0.1, 0] / 0.4
    np.testing.assert_allclose(residual(P, Q), [0.75, 0.25, 0.0])


def test_one_step_chunk_distribution_by_hand():
    # gamma = 1, i.i.d. models. Draft x accepted w.p. min(q_x, p_x) = 0.2 for every x,
    # then a bonus y ~ p:  P((x, y)) = 0.2 * p_y.
    # Rejection mass = 1 - alpha = 0.4, residual r = [.75, .25, 0]:
    #   P((0,)) = 0.3, P((1,)) = 0.1, P((2,)) = 0.
    d = exact_step_distribution(lambda c: P, lambda c: Q, (), gamma=1)
    for x in range(3):
        for y in range(3):
            assert d[(x, y)] == pytest.approx(0.2 * P[y])
    assert d[(0,)] == pytest.approx(0.3)
    assert d[(1,)] == pytest.approx(0.1)
    assert (2,) not in d
    assert sum(d.values()) == pytest.approx(1.0)


def test_first_token_is_exactly_target_distributed():
    # P(first = x) = min(p,q)(x) + (1-alpha) r(x) = p(x), for any p, q.
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.dirichlet(np.ones(5) * 0.7)
        q = rng.dirichlet(np.ones(5) * 0.7)
        q[rng.integers(5)] = 0.0          # draft can miss tokens the target likes
        q /= q.sum()
        d = exact_step_distribution(lambda c: p, lambda c: q, (), gamma=3)
        first = np.zeros(5)
        for chunk, m in d.items():
            first[chunk[0]] += m
        np.testing.assert_allclose(first, p, atol=1e-13)


# ------------------------------------------------------- exactness, whole seqs

def _markov(V, order, seed, sparsity=0.0):
    rng = np.random.default_rng(seed)
    table = {}

    def fn(ctx):
        key = ctx[-order:] if order else ()
        if key not in table:
            v = rng.dirichlet(np.ones(V) * 0.6)
            if sparsity:
                v[rng.random(V) < sparsity] = 0.0
                if v.sum() == 0:
                    v[rng.integers(V)] = 1.0
                v /= v.sum()
            table[key] = v
        return table[key]
    return fn


@pytest.mark.parametrize("gamma", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_sequence_law_equals_target_law(gamma, temperature):
    """Exact enumeration: speculative output law == target autoregressive law."""
    V, L = 3, 4
    target = _markov(V, order=2, seed=10, sparsity=0.25)   # 2nd-order chain, some zeros
    draft = _markov(V, order=1, seed=99, sparsity=0.25)    # wrong order, own zeros
    prefix = (0, 1)
    spec = exact_sequence_distribution(target, draft, prefix, L, gamma, temperature)
    ref = target_sequence_distribution(target, prefix, L, temperature)
    keys = set(spec) | set(ref)
    err = max(abs(spec.get(k, 0.0) - ref.get(k, 0.0)) for k in keys)
    assert err < 1e-12
    assert sum(spec.values()) == pytest.approx(1.0, abs=1e-12)


def test_identical_models_accept_everything():
    f = _markov(4, 1, seed=3)
    d = exact_step_distribution(f, f, (2,), gamma=4)
    assert all(len(k) == 5 for k in d)          # gamma + bonus, always


def test_disjoint_support_rejects_first_draft_always():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert acceptance_probability(p, q) == 0.0
    d = exact_step_distribution(lambda c: p, lambda c: q, (), gamma=3)
    assert d == {(0,): pytest.approx(1.0)}


# ------------------------------------------------------------ closed forms

def test_expected_tokens_closed_form_vs_geometric_sum():
    for a in [0.0, 0.3, 0.6, 0.9, 0.999]:
        for g in range(0, 8):
            assert expected_tokens_per_iteration(a, g) == pytest.approx(sum(a ** k for k in range(g + 1)))
    assert expected_tokens_per_iteration(1.0, 4) == 5.0


def test_expected_tokens_matches_exact_enumeration_iid():
    # With context-free p, q, alpha is constant and the closed form is exact.
    for g in range(0, 6):
        d = exact_step_distribution(lambda c: P, lambda c: Q, (), gamma=g)
        mean = sum(len(k) * v for k, v in d.items())
        assert mean == pytest.approx(expected_tokens_per_iteration(0.6, g), abs=1e-12)


def test_chunk_pmf():
    pmf = chunk_length_pmf(0.6, 3)
    # n=1: .4 ; n=2: .6*.4=.24 ; n=3: .36*.4=.144 ; n=4: .216
    np.testing.assert_allclose(pmf[1:], [0.4, 0.24, 0.144, 0.216])
    assert pmf.sum() == pytest.approx(1.0)


def test_walltime_improvement_and_optimal_gamma():
    assert walltime_improvement(0.6, 1, 0.1) == pytest.approx(1.6 / 1.1)
    assert walltime_improvement(0.0, 3, 0.2) == pytest.approx(1 / 1.6)  # pure overhead
    assert optimal_gamma(0.0, 0.05) == 0                                 # never speculate
    gs = [optimal_gamma(a, 0.05) for a in np.linspace(0.05, 0.95, 19)]
    assert gs == sorted(gs)                                              # monotone in alpha
    assert optimal_gamma(0.95, 0.01) > optimal_gamma(0.95, 0.2)          # cheaper draft -> look further


# ------------------------------------------------------------ Monte Carlo

def test_mc_chunk_lengths_follow_truncated_geometric():
    rng = np.random.default_rng(7)
    t, d = TableLM(lambda c: P), TableLM(lambda c: Q)
    g, n = 3, 40_000
    lengths = np.array([len(speculative_step(t, d, [], g, rng)) for _ in range(n)])
    pmf = chunk_length_pmf(0.6, g)
    for k in range(1, g + 2):
        freq = (lengths == k).mean()
        se = math.sqrt(pmf[k] * (1 - pmf[k]) / n)
        assert abs(freq - pmf[k]) < 5 * se
    assert t.calls == n                        # one target pass per iteration
    assert d.calls == n * g                    # gamma draft passes per iteration


@pytest.fixture(scope="module")
def trained():
    text = generate_corpus(3000, seed=0)
    vocab = Vocab(text)
    ids = vocab.encode(text)
    split = int(0.9 * len(ids))
    target = NGramLM(vocab, order=4).fit(ids[:split])
    draft = NGramLM(vocab, order=1).fit(ids[:split])
    return vocab, ids[split:], target, draft


def test_mc_two_token_marginal_on_trained_models(trained):
    vocab, _, target, draft = trained
    prefix = vocab.encode("the ")
    ref = target_sequence_distribution(target._dist_uncounted, prefix, 2)
    rng = np.random.default_rng(11)
    n = 30_000
    counts: dict = {}
    for _ in range(n):
        out, _ = speculative_generate(target, draft, prefix, 2, gamma=4, rng=rng)
        counts[tuple(out)] = counts.get(tuple(out), 0) + 1
    for k, pk in ref.items():
        if pk < 1e-3:
            continue
        se = math.sqrt(pk * (1 - pk) / n)
        assert abs(counts.get(k, 0) / n - pk) < 5 * se, (vocab.decode(k), pk)


# --------------------------------------------------------------- greedy

@pytest.mark.parametrize("gamma", [1, 3, 6])
def test_greedy_speculative_equals_greedy_target(trained, gamma):
    vocab, _, target, draft = trained
    for seed_text in ["the ", "a clever ", "every robot "]:
        prefix = vocab.encode(seed_text)
        ref = autoregressive_generate(target, prefix, 60, np.random.default_rng(0), temperature=0)
        spec, st = speculative_generate(target, draft, prefix, 60, gamma,
                                        np.random.default_rng(0), temperature=0)
        assert spec == ref
        assert st.iterations < 60                       # it actually saved target passes


# --------------------------------------------------------------- models

def test_ngram_distributions_are_proper_and_full_support(trained):
    vocab, held, target, draft = trained
    for t in range(0, 400, 7):
        for m in (target, draft):
            p = m._dist_uncounted(held[:t])
            assert p.sum() == pytest.approx(1.0)
            assert (p > 0).all()


def test_higher_order_target_has_lower_heldout_perplexity(trained):
    vocab, held, target, draft = trained
    pt, pd = target.perplexity(held), draft.perplexity(held)
    assert pt < pd
    assert pt < len(vocab)                     # beats uniform


def test_batched_verify_counts_one_call(trained):
    vocab, _, target, _ = trained
    before = target.calls
    out = target.dist_batch(vocab.encode("the "), vocab.encode("cat"))
    assert len(out) == 4 and target.calls == before + 1
    np.testing.assert_allclose(out[3], target._dist_uncounted(vocab.encode("the cat")))


def test_gamma_zero_is_plain_sampling():
    rng = np.random.default_rng(0)
    t, d = TableLM(lambda c: P), TableLM(lambda c: Q)
    st = DecodeStats()
    for _ in range(100):
        assert len(speculative_step(t, d, [], 0, rng, stats=st)) == 1
    assert d.calls == 0 and st.tokens_per_iteration == 1.0


def test_temperature():
    np.testing.assert_allclose(apply_temperature(P, 0), [1, 0, 0])
    np.testing.assert_allclose(apply_temperature(P, 1), P)
    sharp = apply_temperature(P, 0.5)                   # p^2 normalised
    np.testing.assert_allclose(sharp, P ** 2 / (P ** 2).sum())
