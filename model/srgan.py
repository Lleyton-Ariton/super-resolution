import torch
import torch.nn as nn

import numpy as np


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int=1, padding: int=1, bias: bool=False):
        super().__init__()
        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias
        }

        self.conv_block = nn.Sequential(
            nn.Conv2d(**self.params),
            nn.BatchNorm2d(self.params['out_channels']),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int=1, padding: int=1, bias: bool=False, upscale_factor: int=2):
        super().__init__()

        self.params = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'bias': bias
        }

        self.upscale_factor = upscale_factor

        self.upsampling_block = nn.Sequential(
            nn.Conv2d(**self.params),
            nn.BatchNorm2d(self.params['out_channels']),
            nn.PixelShuffle(upscale_factor=self.upscale_factor),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsampling_block(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.res_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),

            nn.PReLU(),

            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.res_block(x)

        return x


class GeneratorNetwork(nn.Module):

    def __init__(self, upscale_factor: int=4, num_res_blocks: int=16):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.num_res_blocks = num_res_blocks
        self.num_upsampling_blocks = int(np.log2(self.upscale_factor))

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=False),
            nn.PReLU()
        )

        self.residual_body = nn.Sequential(*[ResidualBlock(64) for _ in range(self.num_res_blocks)])
        self.intermediate = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.upsampling_body = nn.Sequential(
            *[UpsamplingBlock(64, 256, kernel_size=3) for _ in range(self.num_upsampling_blocks)]
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x += self.intermediate(self.residual_body(x))

        x = self.upsampling_body(x)
        x = self.output_layer(x)

        return x


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        pass  # Not Finished
