import math
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class UNetStyleGAN(nn.Module):
    def __init__(
        self, image_resolution: int, in_channels: int, latent_dim: int
    ) -> None:
        super(UNetStyleGAN, self).__init__()
        self.log_resolution = int(math.log(image_resolution, 2))
        self.depth = self.log_resolution - 2

        features = [
            min(256, 32 * (2**i)) for i in range(self.log_resolution - 3, -1, -1)
        ]

        self.num_blocks = len(features)

        self.down_blocks = nn.ModuleList()
        for i in range(self.depth):
            out_channels = features[i]
            self.down_blocks.append(
                UNetDownBlock(in_channels=in_channels, out_channels=out_channels)
            )
            in_channels = out_channels

        self.first_style_block = UNetStyleBlock(latent_dim, in_channels, features[-1])
        self.first_toRGB = ToRGB(latent_dim, features[-1])

        self.generator_blocks = nn.ModuleList(
            [
                UNetGeneratorBlock(latent_dim, features[-i - 1], features[-i - 1])
                for i in range(1, self.num_blocks)
            ]
        )
        self.multichannel_upsample = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear"),
                    nn.Conv2d(
                        in_channels=features[-i],
                        out_channels=features[-i - 1],
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(num_features=features[-i - 1]),
                    nn.ReLU(inplace=True),
                )
                for i in range(1, self.num_blocks)
            ]
        )

        self.upsample = UpSample()
        self.final_convolution = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=3,
                    padding=1,
                ),
            )
        )

    def forward(self, masked_image: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = torch.swapaxes(w, 0, 1)

        skip_connections = []
        for _, down in enumerate(self.down_blocks):
            masked_image = down(masked_image)
            skip_connections.append(masked_image)

        masked_image = self.first_style_block(masked_image, w[0], skip_connections[-1])
        rgb = self.first_toRGB(masked_image, w[0])

        for i in range(1, self.num_blocks):
            masked_image = self.multichannel_upsample[i - 1](masked_image)

            masked_image, rgb_new = self.generator_blocks[i - 1](
                masked_image, w[i], skip_connections[-i - 1]
            )

            rgb = self.upsample(rgb) + rgb_new

        return self.final_convolution(rgb)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UNetDownBlock, self).__init__()

        self.convolution = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.convolution(x) + self.residual(x)
        return self.activation(x)


class UNetGeneratorBlock(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int, out_channels: int):
        super().__init__()

        self.style_block1 = UNetStyleBlock(latent_dim, in_channels, out_channels)
        self.style_block2 = UNetStyleBlock(latent_dim, out_channels, out_channels)
        self.to_rgb = ToRGB(latent_dim, out_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor, skip: torch.Tensor):
        """
        Args:
            x (torch.Tensor): The input image to transform.
                Requires shape to be [batch_size, in_channels, height, width]
            w (torch.Tensor): The style information to insert to the block.
                Requires shape to be [batch_size, latent_dim]
            skip (torch.Tensor): The skip connection from the U-Net encoder.
                Requires shape to be [batch_size, in_channels, height, width]
        """
        x = self.style_block1(x, w, skip)
        x = self.style_block2(x, w, skip)
        rgb = self.to_rgb(x, w)
        return x, rgb


class UNetStyleBlock(nn.Module):
    def __init__(self, latent_dim: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(latent_dim, in_features, bias=1.0)
        self.weight_modulated_conv = Conv2dWeightModulate(
            in_features, out_features, kernel_size=3
        )
        self.scale_image = nn.Parameter(torch.Tensor([0.3]))

        self.convolutional = nn.Sequential(
            nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_features),
            nn.ReLU(inplace=True),
        )

        self.final_activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, skip: torch.Tensor):
        """
        Args:
            x (torch.Tensor): The input image to transform.
                Requires shape to be [batch_size, in_channels, height, width]
            w (torch.Tensor): The style information to insert to the block.
                Requires shape to be [batch_size, latent_dim]
            skip (torch.Tensor): The skip connection from the U-Net encoder.
                Requires shape to be [batch_size, in_channels, height, width]
        """
        style_vector = self.to_style(w)

        x = self.weight_modulated_conv(x, style_vector)

        skip = self.convolutional(skip)

        x = (1 - self.scale_image) * x + self.scale_image * skip

        return self.final_activation(x)


class ToRGB(nn.Module):
    """
    Generate RGB image from feature map with pointwise convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation

    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        demodulate: float = True,
        eps: float = 1e-8,
    ):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt(
                (weights**2).sum(dim=(2, 3, 4), keepdim=True) + self.eps
            )
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)


class UpSample(nn.Module):
    """
    <a id="up_sample"></a>

    ### Up-sample

    The up-sample operation scales the image up by $2 \times$ and [smoothens](#smooth) each feature channel.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    """
    <a id="smooth"></a>

    ### Smoothing Layer

    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    <a id="equalized_conv2d"></a>

    ## Learning-rate Equalized 2D Convolution Layer

    This uses [learning-rate equalized weights](#equalized_weights) for a convolution layer.
    """

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int, padding: int = 0
    ):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights](#equalized_weights)
        self.weight = EqualizedWeight(
            [out_features, in_features, kernel_size, kernel_size]
        )
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c
