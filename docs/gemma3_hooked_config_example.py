"""Utilities for building a HookedTransformerConfig for Gemma 3.

This module is intended as documentation support. It shows how you can
convert an existing Hugging Face Gemma 3 configuration dictionary into the
``HookedTransformerConfig`` object that ``ReplacementModel`` expects.

The helper intentionally keeps the logic explicit so you can tweak it if your
checkpoint deviates (for example, different attention windows or rotary
scaling behaviour).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Mapping

import torch
from transformer_lens import HookedTransformerConfig


def _infer_attn_types(
    hf_config: Mapping[str, object],
    n_layers: int,
    *,
    sliding_window: int | None,
) -> list[str]:
    """Infer the per-layer attention types expected by HookedTransformer."""

    attn_types = hf_config.get("attention_types")
    if isinstance(attn_types, Sequence) and not isinstance(attn_types, (str, bytes)):
        return [str(kind) for kind in attn_types]

    if sliding_window:
        # Default Gemma 2/3 behaviour alternates global and local layers.
        pattern = ("global", "local")
        return [pattern[layer % len(pattern)] for layer in range(n_layers)]

    return ["global"] * n_layers


def build_gemma3_hooked_config(
    hf_config: Mapping[str, object],
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> HookedTransformerConfig:
    """Create a ``HookedTransformerConfig`` that matches a Gemma 3 checkpoint."""

    hidden_size = int(hf_config["hidden_size"])
    n_layers = int(hf_config["num_hidden_layers"])
    n_heads = int(hf_config["num_attention_heads"])
    d_head = hidden_size // n_heads

    sliding_window = hf_config.get("sliding_window")
    if isinstance(sliding_window, Sequence):
        sliding_window = min(int(x) for x in sliding_window)
    elif sliding_window is not None:
        sliding_window = int(sliding_window)

    rope_scaling = hf_config.get("rope_scaling") or {}
    ntk_by_parts = rope_scaling.get("type") == "ntk_by_parts"

    attn_types = _infer_attn_types(hf_config, n_layers, sliding_window=sliding_window)

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=hidden_size,
        n_ctx=int(hf_config["max_position_embeddings"]),
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=int(hf_config["intermediate_size"]),
        act_fn="gelu_pytorch_tanh",
        d_vocab=int(hf_config["vocab_size"]),
        eps=float(hf_config.get("rms_norm_eps", 1e-5)),
        use_attn_result=False,
        use_attn_scale=True,
        attn_scale=math.sqrt(d_head),
        use_split_qkv_input=False,
        use_hook_mlp_in=False,
        use_attn_in=False,
        use_local_attn=sliding_window is not None,
        window_size=sliding_window,
        attn_types=attn_types,
        initializer_range=float(hf_config.get("initializer_range", 0.02)),
        init_mode="gpt2",
        normalization_type="RMSPre",
        device=str(device),
        n_devices=1,
        attention_dir="causal",
        attn_only=False,
        seed=None,
        init_weights=False,
        scale_attn_by_inverse_layer_idx=bool(
            hf_config.get("scale_attn_by_inverse_layer_idx", False)
        ),
        positional_embedding_type="rotary",
        final_rms=True,
        d_vocab_out=int(hf_config.get("vocab_size", hf_config["vocab_size"])),
        parallel_attn_mlp=False,
        rotary_dim=int(hf_config.get("rope_dimension", d_head)),
        use_hook_tokens=False,
        default_prepend_bos=True,
        dtype=dtype,
        tokenizer_name=hf_config.get("tokenizer_name", "google/gemma-3"),
        tokenizer_prepends_bos=True,
        n_key_value_heads=int(hf_config.get("num_key_value_heads", n_heads)),
        post_embedding_ln=bool(hf_config.get("post_attention_norm", False)),
        rotary_base=float(hf_config.get("rope_theta", 10000.0)),
        trust_remote_code=False,
        rotary_adjacent_pairs=bool(hf_config.get("rope_use_alibi", False)),
        load_in_4bit=False,
        num_experts=hf_config.get("num_experts"),
        experts_per_token=hf_config.get("experts_per_token"),
        relative_attention_max_distance=None,
        relative_attention_num_buckets=None,
        decoder_start_token_id=None,
        tie_word_embeddings=bool(hf_config.get("tie_word_embeddings", False)),
        use_normalization_before_and_after=True,
        attn_scores_soft_cap=float(hf_config.get("attn_logit_softcapping", 50.0)),
        output_logits_soft_cap=float(hf_config.get("logits_soft_capping", 0.0)),
        use_NTK_by_parts_rope=ntk_by_parts,
        NTK_by_parts_low_freq_factor=float(
            rope_scaling.get("low_freq_factor", 1.0)
        ),
        NTK_by_parts_high_freq_factor=float(
            rope_scaling.get("high_freq_factor", 4.0)
        ),
        NTK_by_parts_factor=float(rope_scaling.get("factor", 8.0)),
        gated_mlp=True,
        model_name=str(hf_config.get("_name_or_path", "gemma-3")),
        original_architecture=str(
            (hf_config.get("architectures") or ["Gemma3ForCausalLM"])[0]
        ),
        from_checkpoint=False,
        checkpoint_index=None,
        checkpoint_label_type=None,
        checkpoint_value=None,
    )

    return cfg


__all__ = ["build_gemma3_hooked_config"]
