#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2-T2V-A14B to a calibrated Quark MXFP4 checkpoint (offline, Route A).

Produces a diffusers-style transformer directory whose weights are SmoothQuant-
calibrated and MXFP4-rounded (bf16 tensors carrying FP4-grid values + a
quantization config marking the checkpoint serialized). vllm-omni's ROCm offline
loader (ROCmMxfp4OfflineLinearMethod, gfx950) reads it, packs to the AITER FP4
layout at load, and runs the same gemm_a4w4 as the online path - only the weights
are calibrated.

Why "Route A" (float weights, not packed FP4 on disk): Quark's export_safetensors
packing path is Transformers-only; for a bare DiT we keep the frozen bf16 weights
(which already hold the calibrated FP4-rounded values) and let the loader pack at
load. This delivers the calibration accuracy the uncalibrated online path cannot.

Rotation (R1/R2) is intentionally OFF: Quark's RotationProcessor needs a decoder-
shaped scaling dict that does not map cleanly onto Wan's DiT (follow-up). SmoothQuant
uses the per-model scaling map in quark_mxfp4_scaling_maps.py.

The A14B cascade has TWO transformers (transformer + transformer_2); both are
quantized with the same config.

Example:
    python examples/quantization/quantize_wan2_2_quark_mxfp4.py \\
        --model /path/to/Wan2.2-T2V-A14B-Diffusers \\
        --output /path/to/wan2.2-t2v-a14b-quark-mxfp4 \\
        --n-prompts 4 --n-steps 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

from quark_mxfp4_scaling_maps import get_decoder_layers_attr, get_scaling_map

DEFAULT_CALIB_PROMPTS = [
    "A serene lakeside sunrise with mist over the water.",
    "A bustling city street at night with neon signs and rain.",
    "A close-up of a hummingbird hovering over a red flower.",
    "Aerial view of ocean waves crashing on a rocky coast.",
    "A cat walking through tall grass in a sunny meadow.",
    "Time-lapse of clouds moving over snowy mountain peaks.",
]

# Keys Quark adds that vllm-omni's WanTransformer3DModel does not expect. The offline
# loader re-derives the per-32 scale from the calibrated weight via AITER, so these
# are redundant and are stripped from the saved checkpoint.
_DROP_KEY_MARKERS = ("_weight_quantizer", "_input_quantizer", "._amax")


class _ToSafe:
    """Make a non-tensor value survive Quark's cache_model_inps `.to(device)` call.

    Quark's calib input-caching (built for LLMs whose block kwargs are all tensors)
    calls .to(device) on EVERY value. Wan DiT blocks receive non-tensor kwargs
    (bool/None/int); wrapping them so .to() returns the original primitive lets the
    block forward receive the real value.
    """

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def to(self, *a, **k):
        return self.v


class _SafeCalibLoader:
    """Re-iterable wrapper making non-tensor calib kwargs .to()-safe."""

    def __init__(self, dl):
        self._dl = dl

    def __len__(self):
        return len(self._dl)

    def __iter__(self):
        for sample in self._dl:
            if isinstance(sample, dict):
                yield {k: (v if isinstance(v, torch.Tensor) else _ToSafe(v))
                       for k, v in sample.items()}
            else:
                yield sample


def build_qconfig(model, alpha: float, smooth: bool, r1: bool, r2: bool):
    from quark.torch.quantization.config.config import (
        QConfig, QLayerConfig, OCP_MXFP4Spec, RotationConfig, SmoothQuantConfig,
    )

    scaling_layers = get_scaling_map(model)
    decoder_layers = get_decoder_layers_attr(model)

    w_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    global_cfg = QLayerConfig(weight=w_spec)

    algo = []
    if r1 or r2:
        # Fused Hadamard (online_r1_rotation OFF -> folds into weights, no forward op).
        algo.append(RotationConfig(
            scaling_layers=scaling_layers, model_decoder_layers=decoder_layers,
            r1=r1, r2=r2, r3=False, r4=False, online_r1_rotation=False,
        ))
    if smooth:
        algo.append(SmoothQuantConfig(
            scaling_layers=scaling_layers, model_decoder_layers=decoder_layers, alpha=alpha,
        ))
    return QConfig(global_quant_config=global_cfg, algo_config=algo or None)


def quantize_component(args, comp: str) -> dict:
    from diffusers import WanPipeline
    from quark.torch import ModelQuantizer
    from quark.torch.utils.diffusers.calibration import get_calib_dataloader
    from safetensors.torch import save_file

    dev = "cuda"
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tf = getattr(pipe, comp)
    print(f"[quark-mxfp4] {comp}: {type(tf).__name__} loaded in {time.time() - t0:.0f}s")

    qconfig = build_qconfig(tf, args.alpha, not args.no_smooth, args.r1, args.r2)
    pipe.to(dev)
    prompts = DEFAULT_CALIB_PROMPTS[:args.n_prompts]
    dl = _SafeCalibLoader(get_calib_dataloader(pipe, tf, prompts, n_steps=args.n_steps, device=dev))

    quantizer = ModelQuantizer(qconfig)
    tq = quantizer.freeze(quantizer.quantize_model(tf, dataloader=dl))
    dt = time.time() - t0

    out = os.path.join(args.output, comp)
    os.makedirs(out, exist_ok=True)
    sd, dropped = {}, 0
    for k, v in tq.state_dict().items():
        if not isinstance(v, torch.Tensor):
            continue
        if any(m in k for m in _DROP_KEY_MARKERS):
            dropped += 1
            continue
        sd[k] = v.to(torch.bfloat16).contiguous() if v.is_floating_point() else v.contiguous()
    save_file(sd, os.path.join(out, "diffusion_pytorch_model.safetensors"))
    # config.json stanza so vllm-omni auto-selects the offline MXFP4 loader.
    json.dump({"quantization_config": {
        "quant_method": "quark", "quark_export_format": "mxfp4",
        "is_checkpoint_mxfp4_serialized": True, "producer": "quark",
        "algo": {"smoothquant": not args.no_smooth, "alpha": args.alpha,
                 "rotation_r1": args.r1, "rotation_r2": args.r2}}},
        open(os.path.join(out, "quant_config.json"), "w"), indent=2)
    print(f"[quark-mxfp4] {comp}: saved {len(sd)} tensors (dropped {dropped} quantizer keys) "
          f"-> {out} in {dt:.0f}s")

    del pipe, tf, tq
    gc.collect()
    torch.cuda.empty_cache()
    return {"saved": out, "tensors": len(sd), "dropped": dropped, "seconds": round(dt, 1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a calibrated Quark MXFP4 Wan2.2 checkpoint.")
    p.add_argument("--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                   help="Source BF16 diffusers model path or id.")
    p.add_argument("--output", required=True, help="Export root directory.")
    p.add_argument("--components", nargs="+", default=["transformer", "transformer_2"],
                   help="Transformer components to quantize (A14B cascade = both).")
    p.add_argument("--n-prompts", type=int, default=4, help="Calibration prompts.")
    p.add_argument("--n-steps", type=int, default=4, help="Denoise steps per calib prompt.")
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha.")
    p.add_argument("--no-smooth", action="store_true", help="Disable SmoothQuant (plain MXFP4).")
    p.add_argument("--r1", action="store_true", help="Enable fused Hadamard R1 (experimental on DiT).")
    p.add_argument("--r2", action="store_true", help="Enable fused Hadamard R2 (experimental on DiT).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    summary = {"model": args.model, "output": args.output, "alpha": args.alpha,
               "smoothquant": not args.no_smooth, "r1": args.r1, "r2": args.r2,
               "n_prompts": args.n_prompts, "n_steps": args.n_steps, "components": {}}
    for comp in args.components:
        print(f"\n{'=' * 60}\n[quark-mxfp4] component: {comp}\n{'=' * 60}")
        summary["components"][comp] = quantize_component(args, comp)
    json.dump(summary, open(os.path.join(args.output, "export_summary.json"), "w"), indent=2)
    print(f"\n[quark-mxfp4] DONE\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
