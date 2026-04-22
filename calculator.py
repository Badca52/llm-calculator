"""VRAM math for LLM inference.

Pure functions; no Qt imports. Call from the UI layer.

Formula components (all three standard across public calculators):

    weights_gpu = (embedding + lm_head + per_layer * gpu_layers) * bpp
    kv_cache_gpu = 2 * n_layers * ctx * n_kv_heads * head_dim * kv_bytes * (gpu_layers / n_layers)
    overhead = ~300 MB runtime (only when any layer is on GPU)

Embedding and lm_head are always on GPU when gpu_layers > 0 because inference
begins/ends there — offloading only transformer blocks is the llama.cpp model.
"""

from __future__ import annotations

from quantization import BPP, KV_CACHE_BYTES


def effective_config(config: dict) -> dict:
    """Unwrap multimodal configs to expose the transformer parameters.

    Multimodal models (Gemma 3/4, Llama 4, Qwen-VL, etc.) nest the text
    transformer config inside a ``text_config`` object.  Callers need
    ``num_hidden_layers`` and friends at the top level, so return the nested
    dict when present, merging a couple of parent-level hints that don't
    always get duplicated into the child.
    """
    if "num_hidden_layers" in config:
        return config
    inner = config.get("text_config")
    if isinstance(inner, dict) and "num_hidden_layers" in inner:
        merged = dict(inner)
        for inherited in ("tie_word_embeddings", "torch_dtype", "dtype"):
            if inherited not in merged and inherited in config:
                merged[inherited] = config[inherited]
        return merged
    return config


def param_count(api_data: dict, config: dict) -> int:
    """Return total parameter count with fallback.

    Prefer the HF API's ``safetensors.total`` (authoritative, includes
    mixed-dtype components).  Fall back to summing the ``parameters`` dict,
    then to an estimate from config dimensions.  The dimension-based estimate
    is coarse but within ~10% for standard transformer architectures.
    """
    safetensors = api_data.get("safetensors") or {}
    total = safetensors.get("total")
    if isinstance(total, int) and total > 0:
        return total
    params_by_dtype = safetensors.get("parameters")
    if params_by_dtype:
        return sum(params_by_dtype.values())

    cfg = effective_config(config)
    h = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    intermediate = cfg.get("intermediate_size", 4 * h)
    return vocab * h + layers * (4 * h * h + 2 * h * intermediate)


def vram_bytes(
    params: int,
    config: dict,
    quant: str,
    ctx: int,
    kv_dtype: str,
    gpu_layers: int,
    kv_on_cpu: bool = False,
) -> float:
    """Estimate VRAM in bytes for the given configuration.

    Args:
        params: Total model parameter count.
        config: HuggingFace config.json dict.
        quant: Key into BPP (e.g. "Q4_K_M", "FP16").
        ctx: Context length in tokens.
        kv_dtype: Key into KV_CACHE_BYTES ("FP16", "Q8", "Q4").
        gpu_layers: Number of transformer layers offloaded to GPU (0 to n_layers).
        kv_on_cpu: When True, the KV cache is held entirely in system RAM
            (llama.cpp ``--no-kv-offload``).  GPU VRAM then covers only
            weights + overhead; the GPU streams KV over PCIe each step at a
            large speed cost.

    Returns:
        Estimated bytes of VRAM required.
    """
    n_layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    n_kv = config.get("num_key_value_heads", config["num_attention_heads"])
    n_heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", hidden // n_heads)
    tied = config.get("tie_word_embeddings", False)

    bpp = BPP[quant]
    kv_bytes_per = KV_CACHE_BYTES[kv_dtype]

    embed_params = vocab * hidden
    lm_head_params = 0 if tied else vocab * hidden
    transformer_params = max(0, params - embed_params - lm_head_params)
    per_layer_params = transformer_params / n_layers if n_layers else 0

    if gpu_layers > 0:
        weights_gpu = (embed_params + lm_head_params) * bpp
        weights_gpu += per_layer_params * gpu_layers * bpp
    else:
        weights_gpu = 0.0

    if kv_on_cpu:
        kv_gpu = 0.0
    else:
        kv_total = 2 * n_layers * ctx * n_kv * head_dim * kv_bytes_per
        kv_gpu = kv_total * (gpu_layers / n_layers) if n_layers else 0.0

    overhead = 300 * 1024 * 1024 if gpu_layers > 0 else 0

    return weights_gpu + kv_gpu + overhead


def vram_gb(
    params: int,
    config: dict,
    quant: str,
    ctx: int,
    kv_dtype: str,
    gpu_layers: int,
    kv_on_cpu: bool = False,
) -> float:
    return vram_bytes(
        params, config, quant, ctx, kv_dtype, gpu_layers, kv_on_cpu,
    ) / (1024 ** 3)


def kv_cache_bytes(config: dict, ctx: int, kv_dtype: str) -> float:
    """Return the full KV cache size in bytes for the given context length."""
    n_layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    n_kv = config.get("num_key_value_heads", config["num_attention_heads"])
    n_heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", hidden // n_heads)
    kv_bytes_per = KV_CACHE_BYTES[kv_dtype]
    return 2 * n_layers * ctx * n_kv * head_dim * kv_bytes_per


def kv_cache_gb(config: dict, ctx: int, kv_dtype: str) -> float:
    return kv_cache_bytes(config, ctx, kv_dtype) / (1024 ** 3)


def cell_status(vram_gb_value: float, budget_gb: float) -> str:
    """Classify a VRAM value against the user's budget.

    Returns:
        "red"    — exceeds budget
        "yellow" — fits but with less than 1 GB of headroom
        "green"  — fits comfortably
    """
    if vram_gb_value > budget_gb:
        return "red"
    if budget_gb - vram_gb_value < 1.0:
        return "yellow"
    return "green"


def max_ctx_for_budget(
    params: int,
    config: dict,
    quant: str,
    kv_dtype: str,
    gpu_layers: int,
    budget_gb: float,
    ctx_candidates: list[int],
) -> int | None:
    """Return the largest context from ctx_candidates that fits in the budget.

    Returns None if even the smallest candidate exceeds the budget.
    """
    fits = [
        c for c in ctx_candidates
        if vram_gb(params, config, quant, c, kv_dtype, gpu_layers) <= budget_gb
    ]
    return max(fits) if fits else None
