# TransformerLens integration for Gemma 3 workflows

This note explains why `circuit-tracer` relies on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) when you adapt it to Gemma 3 models, and sketches the work involved if you want to replace that dependency.

## Why TransformerLens is still required

* **Model wrapper depends on HookedTransformer internals.** `ReplacementModel` subclasses `HookedTransformer` and immediately replaces each block MLP and the unembedding layer with `HookPoint`-aware wrappers, so it can capture residual activations at the Gemma feature hooks and run interventions. 【F:circuit_tracer/replacement_model.py†L11-L112】
* **Attribution flows through TransformerLens hooks.** The attribution context caches forward activations and computes backward scores by registering hooks whose names follow the TransformerLens convention (for example `blocks.{layer}.{feature_input_hook}` and `hook_embed`). Those hooks receive `HookPoint` objects, which are unique to TransformerLens. 【F:circuit_tracer/attribution/context.py†L1-L118】
* **Gemma configs are expressed with HookedTransformerConfig.** The Gemma-specific tests build both “small” and “large” synthetic Gemma models (mirroring local attention, gated MLPs, and rotary settings) through `HookedTransformerConfig.from_dict` before wrapping them in `ReplacementModel`. The same pathway is used to load the published Gemma‑2 checkpoints and would be needed for Gemma 3 configs. 【F:tests/test_attributions_gemma.py†L1-L210】

In short, Gemma 3 inherits the same circuit-tracing surface as Gemma 2: you need TransformerLens to load the architecture, expose the right hook names, and feed activations into the attribution routines.

## What “circumventing” TransformerLens would entail

If you wanted to operate on Gemma 3 without TransformerLens, you would need to replace each of the surfaces above:

1. **Rebuild the hookable model wrapper.** You could reimplement `ReplacementModel` on top of a plain Hugging Face Gemma 3 module, but you would have to add your own hook registry that mirrors `HookPoint` behaviour so the attribution and intervention code keeps working. That includes exposing per-layer hook names like `blocks.{layer}.mlp.hook_in` / `.hook_out`. 【F:circuit_tracer/replacement_model.py†L24-L173】
2. **Port the attribution manager.** The backward-pass attribution logic currently assumes TransformerLens `HookPoint`s. You would need to switch it to PyTorch’s `register_backward_hook` equivalents (or another hook manager) and keep the buffer layout identical so the scoring matrix is still populated correctly. 【F:circuit_tracer/attribution/context.py†L72-L202】
3. **Mirror HookedTransformerConfig.** All Gemma test fixtures and loading paths expect TransformerLens configs. To support Gemma 3, you would either implement your own dataclass exposing the same fields (local attention windows, gated MLP flags, etc.) or translate Hugging Face Gemma 3 configs into the structure `ReplacementModel` expects. 【F:tests/test_attributions_gemma.py†L148-L236】

None of these steps are Gemma-specific, but Gemma 3 will only work once these TransformerLens surfaces are reproduced or replaced.

## Hand-writing a Gemma 3 `HookedTransformerConfig`

You do not need new TransformerLens primitives to experiment with Gemma 3, because
`ReplacementModel` simply forwards the `HookedTransformerConfig` into the
`HookedTransformer` constructor and then rewires the standard hook points (for
example, `blocks.{layer}.mlp.hook_in` / `.hook_out`).【F:circuit_tracer/replacement_model.py†L42-L114】【F:tests/test_attributions_gemma.py†L219-L318】 As long as your
config matches the architecture—number of layers, rotary settings, local
attention windows, etc.—the hook names stay consistent.

Practically, porting a Gemma 3 checkpoint is “moderately involved” rather than
“hard”: you must translate Hugging Face config fields into the `HookedTransformerConfig`
arguments that TransformerLens expects, but you do not need to touch the rest of
the attribution stack. The main sources of friction are:

1. **Attention layout.** Gemma continues to alternate global and local attention
   when a sliding window is present. You must emit an `attn_types` list that
   matches that schedule so TransformerLens splices in the right hook points for
   each block.【F:docs/gemma3_hooked_config_example.py†L20-L42】
2. **Rotary scaling.** Gemma 3 exposes its NTK-by-parts parameters through
   `rope_scaling`. Those values map directly onto the
   `use_NTK_by_parts_rope`/`NTK_by_parts_*` knobs in TransformerLens.【F:docs/gemma3_hooked_config_example.py†L51-L86】
3. **Soft caps and gating.** Gemma’s soft-capped attention and logits, along with
   its gated MLP, correspond to existing TransformerLens flags. Make sure to copy
   them over so the numerical behaviour matches the checkpoint.【F:docs/gemma3_hooked_config_example.py†L65-L86】

To make the mapping concrete, the repository now includes a helper that consumes
the Hugging Face config dictionary and produces a ready-to-use
`HookedTransformerConfig`.【F:docs/gemma3_hooked_config_example.py†L1-L88】 You can
use it as follows:

```python
from transformers import AutoConfig
import torch

from docs.gemma3_hooked_config_example import build_gemma3_hooked_config

hf_cfg = AutoConfig.from_pretrained("google/gemma-3-4b").to_dict()
hooked_cfg = build_gemma3_hooked_config(hf_cfg, device="cuda", dtype=torch.float16)

model = ReplacementModel.from_config(hooked_cfg, transcoder_set)
```

Feel free to adjust the helper if your checkpoint deviates (for example, if it
uses a different global/local pattern or additional expert settings). The
important thing is that the resulting config exposes the same residual-stream
hook surface that `ReplacementModel` already knows how to use.
