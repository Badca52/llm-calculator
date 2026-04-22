"""Quantization constants for VRAM estimation.

Bytes-per-parameter values are effective BPW measured from real GGUF files.
K-quants mix higher-precision tensors (attention, output) with the nominal
bit-width, pushing the effective BPW above what the name suggests.
"""

BPP: dict[str, float] = {
    "FP16":   2.00,
    "Q8_0":   1.06,
    "Q6_K":   0.82,
    "Q5_K_M": 0.73,
    "Q5_K_S": 0.70,
    "Q4_K_M": 0.61,
    "Q4_K_S": 0.57,
    "Q3_K_M": 0.48,
    "Q3_K_S": 0.42,
    "Q2_K":   0.34,
}

DEFAULT_CONTEXTS: list[int] = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

KV_CACHE_BYTES: dict[str, float] = {"FP16": 2, "Q8": 1, "Q4": 0.5}

MAX_MATRIX_COLUMNS = 9


def format_ctx(n: int) -> str:
    if n >= 1_048_576 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return f"{n:,}"


def contexts_for(max_ctx: int | None) -> list[int]:
    """Return a list of context lengths scaled to the model's max.

    Includes powers of two starting at 2048, always includes the model's
    exact max_position_embeddings as the final column, and caps at
    :data:`MAX_MATRIX_COLUMNS` entries (thinning the middle if needed).
    """
    if not isinstance(max_ctx, int) or max_ctx < 2048:
        return list(DEFAULT_CONTEXTS)

    values: list[int] = []
    c = 2048
    while c < max_ctx:
        values.append(c)
        c *= 2
    values.append(max_ctx)

    if len(values) > MAX_MATRIX_COLUMNS:
        # Preserve the smallest (2048) and the exact max; sample the middle.
        first, last = values[0], values[-1]
        middle = values[1:-1]
        step = max(1, len(middle) // (MAX_MATRIX_COLUMNS - 2))
        trimmed = middle[::step][: MAX_MATRIX_COLUMNS - 2]
        values = [first, *trimmed, last]

    return values
