# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A4 Quark MXFP4 online quantization for diffusion transformers (ROCm gfx950).

SCAFFOLD - the Quark weight/scale transform is not implemented yet (see the TODO
in QuarkMxfp4OnlineLinearMethod.process_weights_after_loading). Everything else
(config, platform dispatch, GEMM custom op, forward path) mirrors the working
built-in ``mxfp4`` path in mxfp4_config.py.

Why a separate method instead of reusing ``mxfp4``:
  The built-in ROCmMxfp4OnlineLinearMethod uses AITER's NATIVE MX format
  (aiter.get_hip_quant(per_1x32) + shuffle_weight(layout=(16,16)) + gemm_a4w4).
  "Quark MXFP4" means matching QUARK's MXFP4 scale/packing layout produced by the
  ``amd-quark`` package, which differs from AITER's native layout. This module
  isolates that difference in one place.

Online only: quantize a plain BF16 checkpoint to FP4 at load time (no calibration).
Offline (pre-quantized Quark MXFP4 checkpoints) is intentionally out of scope here.

Register via factory.register_quantization_override("quark_mxfp4", _build_quark_mxfp4)
so this stays out-of-tree friendly (no edit to the _OVERRIDES literal required).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp8_config import (
    MXFPLinearMethodBase,
    _LazyWeightMixin,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionQuarkMXFP4Config(QuantizationConfig):
    """W4A4 Quark MXFP4 online quantization config for diffusion transformers.

    Online only (BF16 checkpoint -> FP4 at load). gfx950 (ROCm) only for now.
    """

    def __init__(self, ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        # Online-only scaffold: no is_checkpoint_*_serialized flag yet.
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # NOTE: "quark_mxfp4" is registered via register_quantization_override(),
        # not present in vLLM's QuantizationMethods enum. Returned as a plain str.
        return "quark_mxfp4"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper") -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionQuarkMXFP4Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if current_omni_platform.is_rocm():
                gcn_arch = torch.cuda.get_device_properties(
                    torch.accelerator.current_device_index()
                ).gcnArchName
                if "gfx950" not in gcn_arch:
                    raise NotImplementedError(
                        f"Quark MXFP4 on ROCm requires gfx950 (MI355X). Detected: {gcn_arch}"
                    )
                return QuarkMxfp4OnlineLinearMethod(self)
            raise NotImplementedError(
                "DiffusionQuarkMXFP4Config (W4A4 Quark MXFP4) is currently only "
                "supported on ROCm (AMD, gfx950)."
            )
        return None


# ---------------------------------------------------------------------------
# GEMM custom op
# ---------------------------------------------------------------------------


def _register_quark_mxfp4_op() -> None:
    """Register vllm_omni::quark_mxfp4_gemm.

    SCAFFOLD: currently delegates to the same AITER a4w4 GEMM as the native path.
    If Quark's packed weight/scale layout is bit-compatible with AITER's
    per_1x32 + bpreshuffle, this op can be dropped in favor of rocm_mxfp4_gemm.
    If not, replace the body with Quark's kernel call.
    """
    import aiter

    @torch.library.custom_op("vllm_omni::quark_mxfp4_gemm", mutates_args=())
    def _quark_mxfp4_gemm(
        a: torch.Tensor,
        w_quant: torch.Tensor,
        w_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # TODO(quark): if Quark activation quant differs, replace this block.
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        a_quant, a_scale = quant_func(a, shuffle=True)
        return aiter.gemm_a4w4(
            a_quant, w_quant, a_scale, w_scale, bpreshuffle=True, bias=bias
        )

    @_quark_mxfp4_gemm.register_fake
    def _quark_mxfp4_gemm_fake(
        a: torch.Tensor,
        w_quant: torch.Tensor,
        w_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        M, _ = a.shape
        N, _ = w_quant.shape
        return torch.empty(M, N, dtype=a.dtype, device=a.device)


# ---------------------------------------------------------------------------
# ROCm Quark MXFP4 online linear method
# ---------------------------------------------------------------------------


class QuarkMxfp4OnlineLinearMethod(_LazyWeightMixin, MXFPLinearMethodBase):
    """ROCm W4A4 Quark MXFP4 online linear method.

    MRO: QuarkMxfp4OnlineLinearMethod -> _LazyWeightMixin -> MXFPLinearMethodBase
      create_weights  : _LazyWeightMixin (meta device + patched loader)
      process_weights : this class (meta -> materialize, then QUARK quant transform)
      apply / ops     : MXFPLinearMethodBase (shared forward skeleton)
    """

    def __init__(self, quant_config: DiffusionQuarkMXFP4Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        if not hasattr(torch.ops.vllm_omni, "quark_mxfp4_gemm"):
            _register_quark_mxfp4_op()

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Materialise from meta device if needed (same pattern as the native online path).
        if layer.weight is not None and layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # -------------------------------------------------------------------
        # TODO(quark): THE core work. Produce Quark's MXFP4 packed weight +
        # scale from the BF16 layer.weight using the amd-quark package, e.g.:
        #
        #   from quark.torch.export.nn.modules... import <mxfp4 packer>
        #   weight_quant, weight_scale = quark_mxfp4_quantize(layer.weight.data)
        #
        # Must match whatever layout quark_mxfp4_gemm() consumes. Until then,
        # this scaffold falls back to AITER's native transform so the path is
        # runnable end-to-end (i.e. identical numerics to --quantization mxfp4).
        # -------------------------------------------------------------------
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        weight_quant, weight_scale = quant_func(layer.weight.data, shuffle=True)
        weight_shuffled = shuffle_weight(weight_quant, layout=(16, 16))

        layer.register_buffer("weight_shuffle", weight_shuffled, persistent=True)
        layer.register_buffer("weight_scale", weight_scale, persistent=True)

        if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
            delattr(layer, "weight")
            layer.register_parameter("weight", None)

        layer._already_called_process_weights_after_loading = True

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Activation quant happens inside the custom op (see quark_mxfp4_gemm).
        return x, None

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.quark_mxfp4_gemm(
            x_q,
            layer.weight_shuffle,
            layer.weight_scale,
            bias,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output
