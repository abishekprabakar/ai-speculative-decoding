"""Speculative decoding from scratch: draft-then-verify sampling that is exact."""

from .models import NGramLM, TableLM, Vocab, generate_corpus
from .speculative import (DecodeStats, apply_temperature, autoregressive_generate,
                          residual, speculative_generate, speculative_step)
from .theory import (acceptance_probability, chunk_length_pmf, exact_sequence_distribution,
                     exact_step_distribution, expected_tokens_per_iteration, optimal_gamma,
                     target_sequence_distribution, total_variation, walltime_improvement)

__all__ = [
    "NGramLM", "TableLM", "Vocab", "generate_corpus",
    "DecodeStats", "apply_temperature", "autoregressive_generate", "residual",
    "speculative_generate", "speculative_step",
    "acceptance_probability", "chunk_length_pmf", "exact_sequence_distribution",
    "exact_step_distribution", "expected_tokens_per_iteration", "optimal_gamma",
    "target_sequence_distribution", "total_variation", "walltime_improvement",
]
