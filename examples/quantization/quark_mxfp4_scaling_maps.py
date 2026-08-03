#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model scaling_layers maps for Quark SmoothQuant (and, later, rotation).

The ONLY per-model artifact needed to cover a new DiT with the offline Quark MXFP4
flow. The export driver (quantize_wan2_2_quark_mxfp4.py) and the vllm-omni offline
loader (ROCmMxfp4OfflineLinearMethod) are model-agnostic; adding a model = add one
entry to SCALING_MAPS + DECODER_LAYERS_ATTR here.

A scaling_layers entry (schema: quark/torch/algorithm/utils/prepare.py) needs:
  prev_op        : module whose OUTPUT weight absorbs the (inverse) scale
  layers         : the linear(s) whose INPUT is scaled (fused on input dim)
  inp            : REQUIRED - the input-feature key (module name) whose captured
                   activation feeds `layers`; usually the first entry of `layers`.
  module2inspect : optional submodule run for the scale search (e.g. attn/ffn).

Structure derived from vllm_omni/diffusion/models/*/*_transformer.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Wan2.2 (WanTransformer3DModel). The A14B cascade uses TWO of these
# (transformer + transformer_2); the map applies to each identically.
# WanTransformerBlock: norm1->attn1(self), norm2->attn2(cross), norm3->ffn.
# ---------------------------------------------------------------------------
WAN_MAP = [
    {"prev_op": "norm1", "layers": ["attn1.to_qkv"], "inp": "attn1.to_qkv", "module2inspect": "attn1"},
    {"prev_op": "attn1.to_qkv", "layers": ["attn1.to_out"], "inp": "attn1.to_out"},
    {"prev_op": "norm2", "layers": ["attn2.to_q"], "inp": "attn2.to_q", "module2inspect": "attn2"},
    {"prev_op": "attn2.to_v", "layers": ["attn2.to_out"], "inp": "attn2.to_out"},
    {"prev_op": "norm3", "layers": ["ffn.net_0"], "inp": "ffn.net_0", "module2inspect": "ffn"},
    {"prev_op": "ffn.net_0", "layers": ["ffn.net_2"], "inp": "ffn.net_2"},
]

# I2V variant adds encoder K/V projections; appended when i2v=True.
WAN_MAP_I2V_EXTRA = [
    {"prev_op": "norm2", "layers": ["attn2.add_k_proj", "attn2.add_v_proj"], "inp": "attn2.add_k_proj"},
]

# Flux - PLACEHOLDER, filled in during the Flux pass. Two block types (dual-stream
# + single-stream) plus a context branch (add_kv_proj / to_add_out) Wan lacks.
FLUX_MAP: list = []  # TODO(flux)

SCALING_MAPS = {
    "WanTransformer3DModel": WAN_MAP,
    "FluxTransformer2DModel": FLUX_MAP,   # placeholder
}

# Attribute path to the ModuleList of transformer blocks. Quark's processors need
# this (model_decoder_layers); diffusers DiTs are not decoder-style (no model.layers).
DECODER_LAYERS_ATTR = {
    "WanTransformer3DModel": "blocks",
    "FluxTransformer2DModel": "transformer_blocks",   # placeholder (verify at Flux pass)
}


def get_decoder_layers_attr(model) -> str:
    name = type(model).__name__
    if name not in DECODER_LAYERS_ATTR:
        raise NotImplementedError(
            f"No decoder-layers attr for {name!r}. Add it to DECODER_LAYERS_ATTR."
        )
    return DECODER_LAYERS_ATTR[name]


def get_scaling_map(model, i2v: bool = False) -> list:
    """Return the scaling_layers map for a diffusion transformer instance.

    Raises NotImplementedError (not KeyError) for unmapped/placeholder models so the
    export script fails loudly with an actionable message.
    """
    name = type(model).__name__
    if name not in SCALING_MAPS:
        raise NotImplementedError(
            f"No Quark scaling_layers map for {name!r}. Add an entry to SCALING_MAPS. "
            f"Known: {list(SCALING_MAPS)}"
        )
    m = list(SCALING_MAPS[name])
    if name == "WanTransformer3DModel" and i2v:
        m = m + WAN_MAP_I2V_EXTRA
    if not m:
        raise NotImplementedError(
            f"Scaling map for {name!r} is a placeholder (empty). Fill it before exporting."
        )
    return m
