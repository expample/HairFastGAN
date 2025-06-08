import torch
import torch.nn.functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Performs upsample, FIR filter, and downsample (all in one step).
    Works fully on CPU and GPU without custom CUDA extensions.
    """

    # Ensure kernel is 2D
    if kernel.ndim == 1:
        kernel = kernel[:, None] * kernel[None, :]
    kernel = kernel.to(input.device, dtype=input.dtype)

    batch, channel, in_h, in_w = input.shape
    input = input.view(-1, 1, in_h, in_w)

    # Upsample
    if up > 1:
        input = F.pad(input, (0, 0, 0, 0, 0, 0, 0, 0))
        input = F.interpolate(input, scale_factor=up, mode="nearest")

    # Padding
    input = F.pad(input, (pad[0], pad[1], pad[0], pad[1]))

    # Filtering (FIR)
    kernel_flip = torch.flip(kernel, [0, 1]).unsqueeze(0).unsqueeze(0)
    input = F.conv2d(input, kernel_flip, stride=1, padding=0)

    # Downsample
    if down > 1:
        input = input[:, :, ::down, ::down]

    out = input.view(batch, channel, input.shape[-2], input.shape[-1])
    return out
