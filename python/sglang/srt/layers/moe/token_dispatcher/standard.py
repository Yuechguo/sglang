from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import get_moe_runner_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    topk_output: TopKOutput

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


class StandardDispatcher(BaseDispatcher):

    def __init__(self, moe_runner_config: MoeRunnerConfig):
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.local_expert_mapping = None if self.enable_flashinfer_cutlass_moe else moe_runner_config.expert_map

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> DispatchOutput:

        if TopKOutputChecker.format_is_standard(topk_output) and self.local_expert_mapping is not None:
            if TopKOutputChecker.format_is_standard(topk_output):
                topk_output = topk_output._replace(
                    topk_ids=self.local_expert_mapping[topk_output.topk_ids]
                )
            elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                raise NotImplementedError()

        return StandardDispatchOutput(
            hidden_states=hidden_states, topk_output=topk_output
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if isinstance(combine_input, StandardCombineInput):
            return combine_input.hidden_states
        else:
            # TODO: this branch should be removed in the future
            assert isinstance(combine_input, torch.Tensor)
            return combine_input

    def set_quant_config(self, quant_config: dict):
        pass
