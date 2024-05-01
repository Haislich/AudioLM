"""This module contains the definition of the SoundStream architecture
as described in https://arxiv.org/pdf/2107.03312"""

from typing import Tuple
from torch import Tensor, nn
import torch.nn.functional as F


# region Encoder
class CausalConv1d(nn.Conv1d):
    """To guarantee real-time inference, all convolutions are causal.
    This means that padding is only applied to the past but not the future
    in both training and offline inference, whereas no padding is
    used in streaming inference
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int],
        stride: int | Tuple[int] = 1,
        padding: str | int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self.causal_padding_size = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # https://github.com/pytorch/pytorch/issues/1333#:~:text=Isn%27t%20it%20enough%20to%20use%20F.pad%20and%20prepend%20the%20dilation%20amount%20to%20the%20sequence%3F
        return super().forward(F.pad(x, (self.causal_padding_size, 0)))


class ResidualUnit(nn.Module):
    """One of the main parts of the encoder block"""

    # Under the assumption that N = in_channels = out_channels,
    # residual units does not alter dimensions.
    def __init__(self, channels, dilation, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                dilation=dilation,
            ),
            nn.ELU(),
            CausalConv1d(in_channels=channels, out_channels=channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.layers(x) + x


class EncoderBlock(nn.Module):
    """Each of the blocks consists of three residual
    units, containing dilated convolutions with dilation rates of 1,
    3, and 9, respectively, followed by a down-sampling layer in
    the form of a strided convolution"""

    # Under the assumption that N = in_channels = out_channels,
    # residual units does not alter dimensions.
    def __init__(self, channels, stride, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            ResidualUnit(channels=channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(channels=channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(channels=channels // 2, dilation=9),
            nn.ELU(),
            CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=2 * stride,
                stride=stride,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.layers(x)


# endregion


# region SoundStream
class SoundStream:
    """The SoundStream model consists of a sequence of three
        building blocks

    - An encoder, which maps the input to a sequence of embeddings
    - which replaces each embedding by the sum of vectors from a set of finite codebooks,
      thus compressing the representation with a target number
      of bits
    - A decoder, which produces a lossy reconstruction from quantized embeddings.

    Its inputs are samples of an R^T dimensional vector x.
    """

    #


# endregion
