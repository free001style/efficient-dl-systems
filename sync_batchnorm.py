import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float):
        world_size = dist.get_world_size()
        N = input.shape[0] * world_size
        local_sum = input.sum(0)
        local_sq_sum = (input ** 2).sum(0)
        local_stats = torch.cat([local_sum, local_sq_sum])
        dist.all_reduce(local_stats)
        mean, sq_mean = local_stats.chunk(2)
        mean, sq_mean = mean / N, sq_mean / N
        var = sq_mean - mean ** 2
        input_norm = (input - mean) / torch.sqrt(var + eps)
        running_mean.mul_(1 - momentum).add_(momentum * mean)
        running_var.mul_(1 - momentum).add_(momentum * var * N / (N - 1))
        ctx.save_for_backward(input_norm, mean, var + eps)
        return input_norm

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        N = grad_output.shape[0] * world_size
        input_norm, mean, var = ctx.saved_tensors
        grad_sum = torch.sum(grad_output, dim=0)
        grad_input_norm_sum = torch.sum(grad_output * input_norm, dim=0)
        local_stats = torch.cat([grad_sum, grad_input_norm_sum])
        dist.all_reduce(local_stats)
        grad_sum, grad_input_norm_sum = local_stats.chunk(2)
        return (N * grad_output - grad_sum - input_norm * grad_input_norm_sum) / (
                torch.sqrt(var) * N), None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum)
        else:
            return (input - self.running_mean) / torch.sqrt(self.running_var + self.eps)
