# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter
import torch.nn.functional as F

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs, get_int_env_var

__all__ = ["QuarkW8A8Fp8"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
# AITER_FP8_PAD_N/K: padding alignment size (e.g., 128 or 256), 0 or unset means no padding
_fp8_pad_n = get_int_env_var("AITER_FP8_PAD_N")
_fp8_pad_k = get_int_env_var("AITER_FP8_PAD_K")
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight


class QuarkW8A8Fp8(QuarkScheme):

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))
        self.is_static_input_scheme: bool = False
        self.input_qscheme: Optional[str] = None
        if input_config is not None:
            self.is_static_input_scheme = not cast(bool, input_config.get("is_dynamic"))
            self.input_qscheme = cast(str, input_config.get("qscheme"))

        self.per_token = (
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"
        )
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.weight_qscheme == "per_tensor":
            if _is_fp8_fnuz:
                input_scale = getattr(layer, "input_scale", None)
                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                max_w_scale = layer.weight_scale
                weight = layer.weight

            max_w_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=max_w_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.weight_qscheme == "per_channel":
            weight = layer.weight

            if _is_fp8_fnuz:
                input_scale = getattr(layer, "input_scale", None)
                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data
            if self.per_token:
                weight_scale = weight_scale.view(-1, 1)
            if _use_aiter:
                # Pad N and K dimensions to alignment before shuffle
                # AITER_FP8_PAD_N/K specifies the alignment (e.g., 128 or 256), 0 means no padding
                N = weight.shape[0]
                K = weight.shape[-1]
                # Pad K to _fp8_pad_k alignment if set (e.g., 128, 256)
                k_pad_size = (_fp8_pad_k - (K % _fp8_pad_k)) % _fp8_pad_k if _fp8_pad_k > 0 else 0
                # Pad N to _fp8_pad_n alignment if set (e.g., 128, 256)
                n_pad_size = (_fp8_pad_n - (N % _fp8_pad_n)) % _fp8_pad_n if _fp8_pad_n > 0 else 0
                # F.pad format: (left, right, top, bottom) for 2D tensor
                # Here: K is dim 1 (right padding), N is dim 0 (bottom padding)
                pad_weight = F.pad(weight, (0, k_pad_size, 0, n_pad_size)) if (k_pad_size > 0 or n_pad_size > 0) else weight

                layer.weight = Parameter(
                    shuffle_weight(pad_weight, (16, 16)), requires_grad=False
                )
                # Also pad weight_scale for N dimension
                # weight_scale shape: (N, 1) for per_token or (N,) otherwise
                if n_pad_size > 0:
                    if weight_scale.dim() == 2:
                        # Shape is (N, 1), pad on dim 0
                        weight_scale = F.pad(weight_scale, (0, 0, 0, n_pad_size), value=1.0)
                    else:
                        # Shape is (N,), pad on dim 0
                        weight_scale = F.pad(weight_scale, (0, n_pad_size), value=1.0)
                # Store original N for output slicing
                layer.original_out_features = N
            else:
                layer.weight = Parameter(weight, requires_grad=False)
                layer.original_out_features = None
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization scheme {self.weight_qscheme}")

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.weight_qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.weight_qscheme == "per_tensor"
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            set_weight_attrs(weight_scale, {"needs_scalar_to_array": True})

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            set_weight_attrs(input_scale, {"needs_scalar_to_array": True})
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.per_token,
        )
        # Slice output to remove N-padding if weight was padded during loading
        if hasattr(layer, 'original_out_features') and layer.original_out_features is not None:
            output = output[..., :layer.original_out_features]
        return output
