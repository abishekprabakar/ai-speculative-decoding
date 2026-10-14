"""End-to-end demo: train a target and a draft model, then decode speculatively.

    python3 demo.py            # prints the tables in the README
    python3 demo.py --plot     # also writes output/gamma_sweep.png (needs matplotlib)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from specdec import (NGramLM, Vocab, autoregressive_generate, expected_tokens_per_iteration,
                     generate_corpus, optimal_gamma, speculative_generate,
                     walltime_improvement)

C_DRAFT = 0.05          # assumed cost of a draft pass relative to a target pass
N_TOKENS = 4000         # tokens generated per configuration
PROMPT = "the clever robot "


def build_models(seed: int = 0):
    text = generate_corpus(3000, seed=seed)
    vocab = Vocab(text)
    ids = vocab.encode(text)
    split = int(0.9 * len(ids))
    train, held = ids[:split], ids[split:]
    target = NGramLM(vocab, order=4).fit(train)
    draft = NGramLM(vocab, order=1).fit(train)
    return vocab, train, held, target, draft


def run(gamma: int, temperature: float, target, draft, prefix, seed: int):
    target.calls = draft.calls = 0
    out, st = speculative_generate(target, draft, prefix, N_TOKENS, gamma,
                                   np.random.default_rng(seed), temperature)
    cost = target.calls + C_DRAFT * draft.calls
    return out, st, target.calls, draft.calls, N_TOKENS / cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    vocab, train, held, target, draft = build_models()
    print(f"vocab: {len(vocab)} chars    held-out: {len(held)} chars")
    print(f"held-out perplexity  target (5-gram): {target.perplexity(held):.3f}"
          f"   draft (bigram): {draft.perplexity(held):.3f}\n")

    prefix = vocab.encode(PROMPT)
    out, st = speculative_generate(target, draft, prefix, 160, 4, np.random.default_rng(3))
    print("sample (gamma=4, T=1.0):")
    print("  " + PROMPT + vocab.decode(out))
    print(f"  {st.iterations} target passes for 160 chars "
          f"({st.tokens_per_iteration:.2f} chars per pass)\n")

    rows = []
    for temperature in (1.0, 0.7):
        print(f"temperature {temperature}   (cost model: draft pass = {C_DRAFT} target pass)")
        print(" gamma  accept  mean_alpha  tok/iter  pred(alpha)  target_calls  speedup  pred_speedup")
        for gamma in (0, 1, 2, 3, 4, 6, 8):
            _, st, tc, dc, speed = run(gamma, temperature, target, draft, prefix, seed=gamma)
            a = st.mean_alpha if gamma else float("nan")
            pred = expected_tokens_per_iteration(a, gamma) if gamma else 1.0
            pred_speed = walltime_improvement(a, gamma, C_DRAFT) if gamma else 1.0
            rows.append((temperature, gamma, st.tokens_per_iteration, pred, speed, pred_speed))
            acc = f"{st.acceptance_rate:6.3f}" if gamma else "     -"
            al = f"{a:10.3f}" if gamma else "         -"
            print(f" {gamma:5d}  {acc}  {al}  {st.tokens_per_iteration:8.3f}  {pred:11.3f}"
                  f"  {tc:12d}  {speed:6.2f}x  {pred_speed:11.2f}x")
        print()

    print("draft quality vs acceptance (gamma=4, T=1.0):")
    print(" draft          held-out ppl  mean_alpha  tok/iter  target_calls  speedup")
    for order in (0, 1, 2, 3):
        d = NGramLM(vocab, order=order).fit(train)
        _, st, tc, dc, speed = run(4, 1.0, target, d, prefix, seed=40 + order)
        print(f" {order + 1}-gram{'':8s} {d.perplexity(held):11.3f}  {st.mean_alpha:10.3f}"
              f"  {st.tokens_per_iteration:8.3f}  {tc:12d}  {speed:6.2f}x")
    print()

    # greedy: identical output, fewer passes
    target.calls = 0
    ref = autoregressive_generate(target, prefix, 400, np.random.default_rng(0), temperature=0)
    ar_calls = target.calls
    spec, st = speculative_generate(target, draft, prefix, 400, 6, np.random.default_rng(0), 0)
    print("greedy (T=0), 400 chars, gamma=6:")
    print(f"  identical to target-only greedy decode: {spec == ref}")
    print(f"  target passes: {ar_calls} autoregressive vs {st.iterations} speculative"
          f"  (acceptance {st.acceptance_rate:.3f})\n")

    print("closed-form optimal gamma at c = 0.05:")
    print("  " + "  ".join(f"alpha={a:.2f}->{optimal_gamma(a, C_DRAFT)}"
                           for a in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("output", exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for temperature, color in ((1.0, "#2b6cb0"), (0.7, "#c05621")):
            r = [x for x in rows if x[0] == temperature]
            g = [x[1] for x in r]
            axes[0].plot(g, [x[2] for x in r], "o-", color=color, label=f"measured T={temperature}")
            axes[0].plot(g, [x[3] for x in r], "--", color=color, alpha=0.6, label=f"(1-a^(g+1))/(1-a) T={temperature}")
            axes[1].plot(g, [x[4] for x in r], "o-", color=color, label=f"measured T={temperature}")
            axes[1].plot(g, [x[5] for x in r], "--", color=color, alpha=0.6, label=f"closed form T={temperature}")
        axes[0].set(xlabel="gamma (draft tokens per iteration)", ylabel="tokens per target pass",
                    title="Tokens emitted per iteration")
        axes[1].set(xlabel="gamma", ylabel=f"speedup (draft cost c={C_DRAFT})",
                    title="Cost-model speedup vs plain sampling")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig("output/gamma_sweep.png", dpi=130)
        print("\nwrote output/gamma_sweep.png")


if __name__ == "__main__":
    main()
