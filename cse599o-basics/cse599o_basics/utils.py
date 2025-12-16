import torch
from torch import Tensor


def cross_entropy_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    assert len(inputs.shape) == 2, f"Expected inputs have shape (batch_size, vocab_size), but got {inputs.shape}"
    assert len(targets.shape) == 1 and targets.shape[0] == inputs.shape[0], (
        f"Expected targets to have shape (batch_size,), matching inputs' batch size {inputs.shape[0]}, but got {targets.shape}"
    )

    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = inputs_max + torch.log(torch.sum(torch.exp(inputs - inputs_max), dim=-1, keepdim=True))
    target_inputs = inputs[torch.arange(inputs.size(0), device=inputs.device), targets]
    return (log_sum_exp - target_inputs).mean()