"""Memory-bounded causal-LM likelihood scoring for MLX models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache

LOG2 = math.log(2.0)
GIB = 2**30


@dataclass(frozen=True)
class MemorySnapshot:
    active_gib: float
    cache_gib: float
    peak_gib: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def memory_snapshot() -> MemorySnapshot:
    return MemorySnapshot(
        active_gib=mx.get_active_memory() / GIB,
        cache_gib=mx.get_cache_memory() / GIB,
        peak_gib=mx.get_peak_memory() / GIB,
    )


def configure_memory_limits(memory_limit_gib: int, cache_limit_mib: int) -> None:
    """Set conservative allocator limits before loading model weights."""

    mx.set_memory_limit(memory_limit_gib * GIB)
    mx.set_cache_limit(cache_limit_mib * 2**20)
    mx.clear_cache()


def extract_logits(output: object) -> mx.array:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, tuple):
        return output[0]
    return output


def score_tokens_streaming(
    model: object,
    tokens: list[int],
    prefix_token_id: int,
    block_tokens: int,
    abort_peak_gib: float,
) -> tuple[float, MemorySnapshot]:
    """Score one sequence while materializing logits for only a tiny token block.

    A causal LM's logits at input position ``t`` predict target position
    ``t + 1``. We therefore stream ``[prefix] + tokens[:-1]`` through a fresh
    KV cache and score ``tokens`` in matching blocks. The largest logit tensor
    is ``1 x block_tokens x vocabulary`` rather than
    ``batch x full_sentence x vocabulary``.
    """

    if not tokens:
        raise ValueError("Cannot score an empty token sequence")
    if block_tokens < 1:
        raise ValueError("block_tokens must be positive")

    input_tokens = [prefix_token_id, *tokens[:-1]]
    cache = make_prompt_cache(model)
    total_bits = 0.0
    peak = memory_snapshot()

    for start in range(0, len(input_tokens), block_tokens):
        stop = min(start + block_tokens, len(input_tokens))
        inputs = mx.array(input_tokens[start:stop], dtype=mx.int32)[None, :]
        targets = mx.array(tokens[start:stop], dtype=mx.int32)[None, :]
        logits = extract_logits(model(inputs, cache=cache)).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        block_bits = (mx.sum(losses) / LOG2).astype(mx.float32)
        mx.eval(block_bits)
        total_bits += float(block_bits.item())

        del inputs, targets, logits, losses, block_bits
        mx.clear_cache()
        current = memory_snapshot()
        if current.peak_gib > peak.peak_gib:
            peak = current
        if current.peak_gib > abort_peak_gib:
            raise MemoryError(
                f"MLX peak {current.peak_gib:.2f} GiB exceeded the fixed "
                f"{abort_peak_gib:.2f} GiB safety ceiling"
            )

    del cache
    mx.clear_cache()
    return total_bits, peak
