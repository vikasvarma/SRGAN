import numpy as np
import torch
import torchvision
import math
from torchvision import models
from torch import nn

#-------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Unit Residual Convolutional Block

    Architecture:
        1. Conv2D
        2. Batch Normalization
        3. Parametric ReLU
        4. Conv2D
        5. Batch Normalization
    """

    def __init__(self, channels):
        """
            TODO 
        """

        super(ResidualBlock, self).__init__()

        # First convolutional block:
        self.conv1 = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = 3,
            padding      = 1,
            padding_mode = 'replicate'
        )
        self.bn1 = nn.BatchNorm2d(num_features = channels)

        # Activation layer:
        self.prelu = nn.PReLU()

        # Second convolutional block:
        self.conv2 = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = 3,
            padding      = 1,
            padding_mode = 'replicate'
        )
        self.bn2 = nn.BatchNorm2d(num_features = channels)

    def forward(self, input):
        """
        """
        
        residue = self.conv1(input)
        residue = self.bn1(residue)
        residue = self.prelu(residue)
        residue = self.conv2(residue)
        residue = self.bn2(residue)

        return residue + input

#-------------------------------------------------------------------------------
class UpsampleBLock(nn.Module):
    """
    Upsampling Convolutional Block

    Architecture:
        1. Conv2D
        2. Pixel Shuffle
        3. Parametric ReLU
    """

    def __init__(self, channels, scale_factor = 2):
        """
            TODO 
        """

        super(UpsampleBLock, self).__init__()

        self.conv = nn.Conv2d(
            channels, 
            channels * scale_factor ** 2, 
            kernel_size = 3, 
            padding = 1,
            padding_mode = 'replicate'
        )
        self.pshuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, input):
        """
            TODO 
        """

        output = self.conv(input)
        output = self.pshuffle(output)
        output = self.prelu(output)
        return output

#-------------------------------------------------------------------------------
class Generator(nn.Module):
    """
    SR Generator Network 
    """

    def __init__(self, num_blocks = 16, scaling_factor = 4):
        """
        TODO
        """
        super(Generator, self).__init__()

        # Preliminary feature extraction conv layer:
        self.conv1  = nn.Conv2d(3, 64, 9, 1, 4, padding_mode = 'replicate')
        self.prelu1 = nn.PReLU()

        # Create residual blocks:
        self.residual_blocks = [ResidualBlock(64) for _ in range(num_blocks)]
        self.residual_blocks = nn.Sequential(*self.residual_blocks)

        # Create terminal convolutional block:
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, padding_mode = 'replicate')
        self.bn2   = nn.BatchNorm2d(num_features = 64)

        # Create scaling layers:
        self.scaling_blocks = [
            UpsampleBLock(64, 2) for _ in range(int(math.log2(scaling_factor)))
        ]
        self.scaling_blocks = nn.Sequential(*self.scaling_blocks)

        # Terminal convolution:
        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4, padding_mode = 'replicate')

    def forward(self, input):
        """
        Forward propagation of Genertor network
        """

        output  = self.prelu1(self.conv1(input))
        residue = output
        output  = self.residual_blocks(output)
        output  = self.bn2(self.conv2(output))
        output  = output + residue
        output  = self.conv3(self.scaling_blocks(output))

        # Scale the output from [-1 1] to [0 1] range
        output  = (output + 1) / 2

        return output

#-------------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    SR Discriminator Network
    """

    def __init__(self, num_blocks = 8):
        """
        """
        super(Discriminator, self).__init__()

        """
        Create the set of convolutional layers:
        Strategy:
            1. Nth convolutional layer halves the output size if N is even.
            2. Odd indexed convolutional blocks produce same output size.
            3. First convolutional block doesn't perform batch norm.
        """

        layers = list()

        for n in range(num_blocks):
            # Compute the number of output channels:
            in_size  = 3 if n == 0 else 64 * ( 2 ** int((n-1)/2) )
            out_size = 64 * ( 2 ** int(n/2) )

            # Convolutional Layer:
            layers.append(nn.Conv2d(
                in_channels  = in_size, 
                out_channels = out_size, 
                kernel_size  = 3, 
                stride       = 1 + int(n % 2), 
                padding      = 1, 
                padding_mode = 'replicate'
            ))

            # NOTE: Skip appending a batch norm layer for the first block:
            if n != 0:
                layers.append(nn.BatchNorm2d(num_features = out_size))

            layers.append(nn.LeakyReLU())

        # Congregate the convolution blocks:
        self.conv_blocks = nn.Sequential(*layers)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this 
        # will have no effect
        self.pooling = nn.AdaptiveAvgPool2d((6, 6))
        
        # Construct the head:
        out_channels = 64 * (2 ** int((num_blocks - 1)/2))
        self.fc1 = nn.Linear(out_channels * 6 * 6, 1024)
        self.lrelu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, input):
        """
        Forward propagation through the Discriminator network.
        """

        batch_size = input.size(0)
        output = self.conv_blocks(input)
        output = self.pooling(output)
        score  = self.fc2(self.lrelu(self.fc1(output.view(batch_size,-1))))

        return score