# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W4A8 quantization for diffusion transformers on ROCm gfx950 (MI355X).

MXFP4 weights (groups of 32 K-elements sharing one ``float8_e8m0fnu`` exponent)
multiplied against MXFP8 activations that are quantized *dynamically* inside the
kernel. Two variants share this config, selected by ``svd_rank``:

  ``svd_rank`` absent   plain      ``y = Q(x) @ Q(W).T + bias``
  ``svd_rank`` present  SVDQuant   ``y = Q(x) @ Q(Wr).T + (x @ L1.T) @ L2.T + bias``

where ``Wr = W - L2 @ L1`` is the 4-bit residual and the low-rank up-projection
is fused into the GEMM epilogue, so both variants are one kernel launch.

Checkpoint contract
-------------------
Two load modes, chosen by ``is_checkpoint_w4a8_serialized``:

**Online (default, stock BF16 checkpoint).** Both variants read a stock BF16
checkpoint and do all their work at load time -- no export step, nothing extra in
the state dict. ``plain`` packs each weight to MXFP4 (RTN) as it loads; ``SVDQuant``
additionally derives ``proj_down`` (R, K) / ``proj_up`` (N, R) from the weight with
``torch.svd_lowrank`` and quantizes only the residual, keeping the factors as
non-persistent buffers. This online SVD is a *weight* SVD, not the activation-aware
smoothing the paper describes (see ``_low_rank_split``).

**Serialized (calibrated checkpoint).** A checkpoint produced offline by
``examples/quantization/export_quark_svdquant_w4a8.py`` (Quark's
``SVDQuantProcessor``: SmoothQuant smoothing + exact SVD on the smoothed weight)
carries the BF16 residual under ``weight`` and the calibrated factors under
``proj_down`` / ``proj_up`` as ordinary checkpoint keys. Self-attention QKV is
pre-fused in the exporter, so a fused ``to_qkv`` layer's factors have rank
``3 * svd_rank``. :class:`QuarkW4A8SVDCheckpointLinearMethod` loads the factors
instead of deriving them; :class:`QuarkW4A8CheckpointLinearMethod` handles the plain
serialized case. The on-disk weights are unpacked BF16 (packed to the kernel layout
at load), because the shuffled MXFP4 layout is kernel-specific rather than a stable
format.

An opt-in ``quark_export_format="mxfp4_packed"`` checkpoint instead stores the
residual already packed (``weight_shuffle``/``weight_scale`` uint8) -- ~4x smaller
and no pack at load, handled by :class:`QuarkW4A8PackedLinearMethod` /
:class:`QuarkW4A8SVDPackedLinearMethod`. This trades portability for size/speed: the
packed layout is tied to the kernel's pack version.

Both variants only support TP=1 today; see ``_reject_tp`` below.
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

from vllm_omni.quantization import flydsl_w4a8
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)
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


class DiffusionQuarkW4A8Config(QuantizationConfig):
    """MXFP4-weight / MXFP8-activation config, with or without SVDQuant.

    Args:
        svd_rank: rank of the low-rank correction branch. ``None`` selects the
            plain variant.
        ignored_layers: layer name patterns to leave in BF16.
        is_checkpoint_w4a8_serialized: load a calibrated serialized checkpoint
            (BF16 residual + on-disk ``proj_down``/``proj_up`` factors) instead of
            quantizing a stock BF16 checkpoint at load. See the module docstring.
        quark_export_format: on-disk residual format of a serialized checkpoint.
            ``"mxfp4_packed"`` = residual pre-packed to the kernel layout
            (``weight_shuffle``/``weight_scale``); ``None``/``"bf16"`` = unpacked
            BF16 residual packed at load.
    """

    def __init__(
        self,
        svd_rank: int | None = None,
        ignored_layers: list[str] | None = None,
        is_checkpoint_w4a8_serialized: bool = False,
        quark_export_format: str | None = None,
    ) -> None:
        super().__init__()
        if svd_rank is not None and svd_rank <= 0:
            raise ValueError(f"svd_rank must be a positive integer, got {svd_rank!r}")
        self.svd_rank = svd_rank
        self.ignored_layers = ignored_layers or []
        self.is_checkpoint_w4a8_serialized = is_checkpoint_w4a8_serialized
        self.quark_export_format = quark_export_format

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # Not a member of vllm's QuantizationMethods Literal; vllm-omni registers
        # out-of-tree method names through the quantization factory, exactly as
        # DiffusionMXFP4DualScaleMixedConfig does for "mxfp4_dualscale".
        return "quark_w4a8"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # The kernel returns bf16 and quantizes its own activations; fp16 in
        # would silently round-trip through bf16.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Only consulted on CUDA. ROCm gating happens in flydsl_w4a8.supports().
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionQuarkW4A8Config:
        svd_rank = cls.get_from_keys_or(config, ["svd_rank", "rank"], None)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_w4a8_serialized"], False)
        export_format = cls.get_from_keys_or(config, ["quark_export_format"], None)
        return cls(
            svd_rank=svd_rank,
            ignored_layers=ignored_layers,
            is_checkpoint_w4a8_serialized=is_serialized,
            quark_export_format=export_format,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        usable, reason = flydsl_w4a8.supports()
        if not usable:
            raise NotImplementedError(f"quantization='quark_w4a8' is unavailable: {reason}")

        in_features, out_features = layer.input_size, layer.output_size

        if self.is_checkpoint_w4a8_serialized:
            # Serialized checkpoint (exported by export_quark_svdquant_w4a8.py): BF16
            # residual + optional low-rank factors on disk, packed to MXFP4 at load.
            # The exporter guarantees factors exist iff the fused shape passes
            # supports_svd_shape, so shape routing here stays consistent with what
            # the checkpoint actually carries.
            packed = self.quark_export_format == "mxfp4_packed"
            if self.svd_rank is not None and flydsl_w4a8.supports_svd_shape(in_features, out_features):
                return (QuarkW4A8SVDPackedLinearMethod if packed else QuarkW4A8SVDCheckpointLinearMethod)(self)
            if flydsl_w4a8.supports_shape(in_features, out_features):
                return (QuarkW4A8PackedLinearMethod if packed else QuarkW4A8CheckpointLinearMethod)(self)
            logger.warning(
                "quark_w4a8 (serialized): %s has shape (out=%d, in=%d), which the kernel "
                "cannot tile; keeping this layer in BF16.",
                prefix,
                out_features,
                in_features,
            )
            return UnquantizedLinearMethod()

        if self.svd_rank is None:
            if not flydsl_w4a8.supports_shape(in_features, out_features):
                logger.warning(
                    "quark_w4a8: %s has shape (out=%d, in=%d), which the W4A8 kernel cannot "
                    "tile; keeping this layer in BF16.",
                    prefix,
                    out_features,
                    in_features,
                )
                return UnquantizedLinearMethod()
            return QuarkW4A8OnlineLinearMethod(self)

        if not flydsl_w4a8.supports_svd_shape(in_features, out_features):
            # Quark refuses these shapes outright rather than emitting garbage
            # (flydsl_svdquant_inference_linear.py:115-126). Wan's proj_out,
            # out=192, is the motivating case.
            logger.warning(
                "quark_w4a8(svd_rank=%d): %s has shape (out=%d, in=%d); the fused SVD epilogue "
                "requires both >= 256 and a multiple of 256. Keeping this layer in BF16.",
                self.svd_rank,
                prefix,
                out_features,
                in_features,
            )
            return UnquantizedLinearMethod()
        return QuarkW4A8SVDOnlineLinearMethod(self)


# ---------------------------------------------------------------------------
# Linear methods
# ---------------------------------------------------------------------------


def _reject_tp(layer: Module, variant: str) -> None:
    """Refuse TP>1 rather than silently sharding the weight wrongly.

    Plain W4A8 shards like any BF16 weight (each rank packs its own slice), but
    it is untested here; the SVD branch additionally has no defined sharding for
    ``proj_down``/``proj_up``, whose rank dimension is replicated.
    """
    tp_size = getattr(layer, "tp_size", 1)
    if tp_size > 1:
        raise NotImplementedError(f"quark_w4a8 ({variant}) has only been validated at TP=1, got tp_size={tp_size}.")


class QuarkW4A8LinearMethod(MXFPLinearMethodBase):
    """Plain W4A8: MXFP4 weight, dynamically quantized MXFP8 activation.

    Load-time buffers, after ``process_weights_after_loading``:

      ``_kernel_weight`` : MXFP4 e2m1, packed and shuffled for the FlyDSL GEMM
      ``_kernel_scale``  : per-group-of-32 E8M0 exponents, same layout

    Both are ``persistent=False``: they are derived from the BF16 checkpoint at
    load and are tied to a specific kernel layout, so they must not leak into a
    ``state_dict``. This mirrors Quark's own inference linear.

    Forward path:
      ``_quantize_activation`` passes through — activation quantization happens
      inside the custom op, so that torch.compile sees one opaque node.
    """

    def __init__(self, quant_config: DiffusionQuarkW4A8Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        flydsl_w4a8.register_ops()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "plain")
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=extra_weight_attrs.get("weight_loader"),
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        packed, scale = flydsl_w4a8.pack_weight(layer.weight.data)
        layer.register_buffer("_kernel_weight", packed, persistent=False)
        layer.register_buffer("_kernel_scale", scale, persistent=False)

        # Drop the BF16 copy immediately; on A14B the two experts are resident
        # at once and the transient doubles peak load memory otherwise.
        if isinstance(getattr(layer, "weight", None), torch.nn.Parameter):
            delattr(layer, "weight")
            layer.register_parameter("weight", None)

        layer._already_called_process_weights_after_loading = True

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Activation quantization to MXFP8 is inside the custom op called by
        # _quant_matmul, so pass the raw activation through here.
        return x, None

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor | None,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.flydsl_w4a8_gemm(
            x_q,
            layer._kernel_weight,
            layer._kernel_scale,
            bias,
            layer.output_size_per_partition,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output


class QuarkW4A8OnlineLinearMethod(_LazyWeightMixin, QuarkW4A8LinearMethod):
    """Plain W4A8 against a stock BF16 checkpoint.

    MRO: QuarkW4A8OnlineLinearMethod -> _LazyWeightMixin -> QuarkW4A8LinearMethod
         -> MXFPLinearMethodBase -> LinearMethodBase

      create_weights  : _LazyWeightMixin  (meta device + patched loader, so the
                        BF16 weight is materialised one layer at a time)
      process_weights : here (meta -> materialise), then the base packs it
      apply / ops     : QuarkW4A8LinearMethod / MXFPLinearMethodBase
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "plain")
        _LazyWeightMixin.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

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

        QuarkW4A8LinearMethod.process_weights_after_loading(self, layer)


class QuarkW4A8SVDLinearMethod(QuarkW4A8LinearMethod):
    """W4A8 plus a rank-R low-rank correction fused into the GEMM epilogue.

    The quantized operand is the residual ``Wr = W - L2 @ L1``; the correction
    is carried by two unquantized BF16 tensors:

      ``proj_down`` : (R, K)  — ``L1``, applied as ``d = x @ L1.T`` in BF16
      ``proj_up``   : (N, R)  — ``L2``, fused into the epilogue as ``d @ L2.T``

    This class holds only the forward path. Where the factors come from is the
    subclass's business, and today the only answer is
    :class:`QuarkW4A8SVDOnlineLinearMethod`, which derives them from the BF16
    weight at load time -- see its docstring for why there is no checkpoint
    variant.
    """

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor | None,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(
            x_q,
            layer._kernel_weight,
            layer._kernel_scale,
            layer.proj_down,
            layer.proj_up,
            bias,
            layer.output_size_per_partition,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output


def _low_rank_split(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``W`` into ``L2 @ L1 + residual`` using its top-``rank`` subspace.

    Returns ``(residual, proj_up, proj_down)`` with ``proj_up`` (N, R) and
    ``proj_down`` (R, K), both carrying ``sqrt(s)`` so neither factor has a
    wildly different dynamic range from the other.

    ``svd_lowrank`` (randomized range finding) rather than a full SVD: only the
    leading R directions are wanted, and a full decomposition of a 13824x5120
    weight per layer would dominate model load time.

    Note this is a *plain* weight SVD, not the activation-aware smoothing that
    the SVDQuant paper describes -- that needs calibration data this path does
    not have. It captures the low-rank term of the method and none of the
    outlier migration, so treat its accuracy as a floor, not the published
    result.
    """
    w = weight.float()
    q = min(rank + 8, *w.shape)
    u, s, v = torch.svd_lowrank(w, q=q, niter=4)
    root = s[:rank].sqrt()
    proj_up = u[:, :rank] * root
    proj_down = root.unsqueeze(1) * v[:, :rank].T
    residual = w - proj_up @ proj_down
    return (
        residual.to(weight.dtype),
        proj_up.to(weight.dtype).contiguous(),
        proj_down.to(weight.dtype).contiguous(),
    )


class QuarkW4A8SVDOnlineLinearMethod(_LazyWeightMixin, QuarkW4A8SVDLinearMethod):
    """SVD-corrected W4A8 against a stock BF16 checkpoint.

    The factors are derived from the weight at load time instead of read from
    disk, for the same reason the plain variant quantizes at load time: no
    exporter emits them. ``proj_down``/``proj_up`` are therefore registered as
    non-persistent *buffers* in ``process_weights_after_loading`` rather than as
    parameters in ``create_weights`` -- if they were parameters they would be
    checkpoint keys that no checkpoint has.

    MRO: -> _LazyWeightMixin -> QuarkW4A8SVDLinearMethod -> QuarkW4A8LinearMethod
         -> MXFPLinearMethodBase -> LinearMethodBase
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "svd")
        _LazyWeightMixin.create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.svd_rank = self.quant_config.svd_rank

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

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

        residual, proj_up, proj_down = _low_rank_split(layer.weight.data, layer.svd_rank)
        layer.register_buffer("proj_down", proj_down, persistent=False)
        layer.register_buffer("proj_up", proj_up, persistent=False)
        layer.weight.data = residual

        QuarkW4A8LinearMethod.process_weights_after_loading(self, layer)


# ---------------------------------------------------------------------------
# Serialized-checkpoint linear methods (offline calibrated export)
# ---------------------------------------------------------------------------


class QuarkW4A8CheckpointLinearMethod(QuarkW4A8OnlineLinearMethod):
    """Plain W4A8 from a serialized checkpoint.

    Mechanically identical to the online plain method -- a BF16 weight is loaded
    one layer at a time and packed to MXFP4 -- the only difference being the
    weight is real calibrated/stock data from disk rather than dummy-initialized.
    Reusing the online method keeps the lazy per-layer pack (no whole-model BF16
    transient at load).
    """


def _swap_param_to_buffer(layer: Module, name: str, tensor: torch.Tensor) -> None:
    if name in layer._parameters:
        del layer._parameters[name]
    layer.register_buffer(name, tensor.contiguous(), persistent=False)


class QuarkW4A8SVDCheckpointLinearMethod(QuarkW4A8SVDLinearMethod):
    """SVD-corrected W4A8 from a serialized checkpoint.

    The residual ``weight`` and the ``proj_down`` / ``proj_up`` factors are all
    read from disk (unlike :class:`QuarkW4A8SVDOnlineLinearMethod`, which derives
    the factors from the weight at load). The exporter pre-fuses self-attention
    QKV, so a fused ``to_qkv`` layer carries rank ``3 * svd_rank``; the per-layer
    rank is recovered from ``len(output_partition_sizes)``.

    Not lazy: all three tensors must be loaded before packing, and their load
    order in the checkpoint is not guaranteed, so packing waits for vLLM's
    post-load ``process_weights_after_loading`` call.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "svd")
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # rank_eff is 3*svd_rank on a fused to_qkv (three output shards) and
        # svd_rank on a single-shard linear -- matching the exporter's fusion.
        # svd_rank is always set here: routing only picks this method when it is.
        assert self.quant_config.svd_rank is not None
        rank_eff = self.quant_config.svd_rank * len(output_partition_sizes)

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "proj_down",
            ModelWeightParameter(
                data=torch.empty(rank_eff, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "proj_up",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, rank_eff, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        proj_down = layer.proj_down.data
        proj_up = layer.proj_up.data
        # Packs the residual into _kernel_weight/_kernel_scale and drops the BF16
        # weight. Deliberately skips _low_rank_split -- the factors are on disk.
        QuarkW4A8LinearMethod.process_weights_after_loading(self, layer)
        _swap_param_to_buffer(layer, "proj_down", proj_down)
        _swap_param_to_buffer(layer, "proj_up", proj_up)
        layer._already_called_process_weights_after_loading = True


# ---------------------------------------------------------------------------
# Pre-packed serialized checkpoints (--pack export: residual already in the
# MXFP4 kernel layout on disk, no packing at load)
# ---------------------------------------------------------------------------


def _register_packed_weight(
    layer: Module, input_size_per_partition: int, output_size_per_partition: int, weight_loader
) -> None:
    """Register the on-disk packed residual: ``weight_shuffle`` (N, K/2) and
    ``weight_scale`` (N, K/32), both uint8.

    K is always a multiple of 256 for a quantized layer (the shape gate enforces
    it), so K/32 is a multiple of 8 and the E8M0 scale needs no padding -- the
    shapes are exact. ``input_dim=None`` keeps the packed K axis unsharded;
    ``output_dim=0`` lets the QKV/MLP shard loaders place row-slices (TP=1 only).
    """
    layer.register_parameter(
        "weight_shuffle",
        ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition // 2, dtype=torch.uint8),
            input_dim=None,
            output_dim=0,
            weight_loader=weight_loader,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        ModelWeightParameter(
            data=torch.empty(output_size_per_partition, input_size_per_partition // 32, dtype=torch.uint8),
            input_dim=None,
            output_dim=0,
            weight_loader=weight_loader,
        ),
    )


def _install_packed_kernel_buffers(layer: Module) -> None:
    """Hand the loaded packed bytes straight to the kernel buffers, no packing.

    The GEMM op already consumes the uint8 views, so ``weight_shuffle`` /
    ``weight_scale`` become ``_kernel_weight`` / ``_kernel_scale`` verbatim.
    """
    kernel_weight = layer.weight_shuffle.data
    kernel_scale = layer.weight_scale.data
    for name in ("weight_shuffle", "weight_scale"):
        if name in layer._parameters:
            del layer._parameters[name]
    layer.register_buffer("_kernel_weight", kernel_weight, persistent=False)
    layer.register_buffer("_kernel_scale", kernel_scale, persistent=False)


class QuarkW4A8PackedLinearMethod(QuarkW4A8LinearMethod):
    """Plain W4A8 from a pre-packed serialized checkpoint (``mxfp4_packed``)."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "plain")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        _register_packed_weight(
            layer, input_size_per_partition, output_size_per_partition, extra_weight_attrs.get("weight_loader")
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        _install_packed_kernel_buffers(layer)
        layer._already_called_process_weights_after_loading = True


class QuarkW4A8SVDPackedLinearMethod(QuarkW4A8SVDLinearMethod):
    """SVD W4A8 from a pre-packed serialized checkpoint: packed residual on disk,
    BF16 ``proj_down`` / ``proj_up`` factors alongside it."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        _reject_tp(layer, "svd")
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        _register_packed_weight(layer, input_size_per_partition, output_size_per_partition, weight_loader)

        assert self.quant_config.svd_rank is not None
        rank_eff = self.quant_config.svd_rank * len(output_partition_sizes)
        layer.register_parameter(
            "proj_down",
            ModelWeightParameter(
                data=torch.empty(rank_eff, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "proj_up",
            ModelWeightParameter(
                data=torch.empty(output_size_per_partition, rank_eff, dtype=params_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        proj_down = layer.proj_down.data
        proj_up = layer.proj_up.data
        _install_packed_kernel_buffers(layer)
        _swap_param_to_buffer(layer, "proj_down", proj_down)
        _swap_param_to_buffer(layer, "proj_up", proj_up)
        layer._already_called_process_weights_after_loading = True
