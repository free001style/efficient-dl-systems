import torch
from torch import nn
import os


class _OffloadedLinearOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, saved_weight_path, bias_or_none):
        weight = torch.load(saved_weight_path, map_location="cpu").pin_memory()
        stream_for_prefetch = torch.cuda.Stream()
        with torch.cuda.stream(stream_for_prefetch):
            weight = weight.to(input.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream_for_prefetch)
        ctx._saved_weight_path = saved_weight_path
        ctx._has_bias = bias_or_none is not None
        return torch.nn.functional.linear(input, weight, bias=bias_or_none)

    @staticmethod
    def backward(ctx, grad_output):
        weight = torch.load(ctx._saved_weight_path, map_location="cpu").to(grad_output.dtype)
        stream_for_prefetch = torch.cuda.Stream()
        with torch.cuda.stream(stream_for_prefetch):
            weight = weight.to(grad_output.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(stream_for_prefetch)
        grad_input = torch.nn.functional.linear(grad_output, weight.t())
        grad_bias = grad_output.flatten(0, -2).sum(0) if ctx._has_bias else None
        return grad_input, None, grad_bias


class OffloadingLinear(nn.Module):
    def __init__(self, weight_path, bias_or_none):
        super().__init__()
        self.saved_weight_path = weight_path
        self.bias_or_none = bias_or_none

    def forward(self, input: torch.Tensor):
        return _OffloadedLinearOp.apply(input, self.saved_weight_path, self.bias_or_none)


def offload_model(model, dir="linear_weight"):
    os.makedirs(dir, exist_ok=True)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lora" not in name:
            weight_path = os.path.join(dir, f"{name}.pth")
            torch.save(module.weight.detach().cpu(), weight_path)
            new_linear = OffloadingLinear(weight_path, module.bias)
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(model.get_submodule(parent_name), child_name, new_linear)
            else:
                setattr(module, name, new_linear)
