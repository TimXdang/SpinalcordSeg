# Definition of different models

import torch
from torch import nn
import segmentation_models_pytorch as smp
from typing import Union, List


class UNet2DRemake:
    """
    2D U-net from https://github.com/qubvel/segmentation_models.pytorch

    :ivar encoder: encoder backbone structure
    :ivar encoder_weights: pre-trained weights
    :ivar activation: final layer activation
    """
    def __init__(self):
        self.encoder = 'resnext101_32x8d'
        self.encoder_weights = 'imagenet'
        self.activation = 'sigmoid'

        self.model = smp.Unet(encoder_name=self.encoder, encoder_weights=self.encoder_weights, classes=1,
                              activation=self.activation, in_channels=1)

    def model(self):
        return self.model()


class Conv3DBlock(nn.Module):
    """
    Encoder block element for 3D U-net. Subclass of torch.nn and returns output and residual link information. It
    applies convolutions, batch normalization, max pooling and uses the ReLU function as activation.

    :ivar conv1: First 3D convolution
    :ivar conv2: Second 3D convolution
    :ivar bn1: First 3D batch normalization
    :ivar bn2: Second 3D batch normalization
    :ivar relu: ReLU activation
    :ivar bottleneck: Indicator if bottleneck
    :ivar pool: 3D max pooling
    """
    def __init__(self, in_channels: int, out_channels: int, bottleneck: bool = False):
        """
        Constructor

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param bottleneck: True if used in bottleneck
        """
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3),
                               padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        """
        Forward pass.

        :param x: input
        :return: output and residual link
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if not self.bottleneck:
            out = self.pool(x)
        else:
            out = x
        return out, x


class UpConv3DBlock(nn.Module):
    """
    Decoder block element for 3D U-net. Subclass of torch.nn and applies upconvolutions, convolutions, batch
    normalization and uses ReLU as an activation.

    :ivar upconv1: Up-convolution
    :ivar conv1: First 3D convolution
    :ivar conv2: Second 3D convolution
    :ivar conv3: Third 3D convolution
    :ivar relu: ReLU activation
    :ivar bn: Batch normalization
    :ivar last: Indicates whether last layer (layer with highest image resolution)
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: Union[int, None] = None, last: bool = False):
        """
        Constructor

        :param in_channels: number of input channels
        :param skip_channels: number of channels coming from skip connection
        :param out_channels: number of output channels
        :param last: indicates if this is a block for the last layer, meaning the layer with the highest image resolution
        """
        super(UpConv3DBlock, self).__init__()
        assert (last is False and out_channels is None) or (last is True and out_channels is not None), \
            'Invalid arguments!'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2),
                                          stride=2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + skip_channels, out_channels=in_channels // 2,
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.last = last
        if last:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=out_channels, kernel_size=(1, 1, 1))

    def forward(self, x, skip=None):
        """
        Forward pass.

        :param x: input
        :param skip: residual connection
        :return: output
        """
        x = self.upconv1(x)
        if skip is not None:
            x = torch.cat((x, skip), 1)
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        if self.last:
            x = self.conv3(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-net according to https://arxiv.org/abs/1606.06650

    :ivar encoder1: High level encoder of U-net
    :ivar encoder2: Middle level encoder of U-net
    :ivar encoder3: Bottom level encoder of U-net
    :ivar bottleneck: Bottleneck convolutions of U-net
    :ivar decoder1: High level decoder of U-net
    :ivar decoder2: Middle level decoder of U-net
    :ivar decoder3: Bottom level decoder of U-net
    """
    def __init__(self, in_channels: int, out_channels: int, lvl_channels: List[int] = [64, 128, 256],
                 bottleneck_channel: int = 512):
        """
        Constructor

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param lvl_channels: number of channels in the 3 defined U-net levels in form of a list
        :param bottleneck_channel: number of channels in the bottleneck
        """
        super(UNet3D, self).__init__()
        lvl1_channels, lvl2_channels, lvl3_channels = lvl_channels
        self.encoder1 = Conv3DBlock(in_channels=in_channels, out_channels=lvl1_channels)
        self.encoder2 = Conv3DBlock(in_channels=lvl1_channels, out_channels=lvl2_channels)
        self.encoder3 = Conv3DBlock(in_channels=lvl2_channels, out_channels=lvl3_channels)
        self.bottleneck = Conv3DBlock(in_channels=lvl3_channels, out_channels=bottleneck_channel, bottleneck=True)
        self.decoder3 = UpConv3DBlock(in_channels=bottleneck_channel, skip_channels=lvl3_channels)
        self.decoder2 = UpConv3DBlock(in_channels=lvl3_channels, skip_channels=lvl2_channels)
        self.decoder1 = UpConv3DBlock(in_channels=lvl2_channels, skip_channels=lvl1_channels, out_channels=out_channels,
                                      last=True)

    def forward(self, x):
        """
        Forward pass.

        :param x: input
        :return: output
        """
        # Encoder
        x, res_lvl1 = self.encoder1(x)
        x, res_lvl2 = self.encoder2(x)
        x, res_lvl3 = self.encoder3(x)
        x, _ = self.bottleneck(x)

        # Decoder
        x = self.decoder3(x, res_lvl3)
        x = self.decoder2(x, res_lvl2)
        x = self.decoder1(x, res_lvl1)

        return x


class Reduced3DUnet(nn.Module):
    def __init__(self, in_channels, out_channels, lvl_channels=[64, 128], bottleneck_channel=256) -> None:
        super(Reduced3DUnet, self).__init__()
        lvl_1_chnls, lvl_2_chnls = lvl_channels[0], lvl_channels[1]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=lvl_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=lvl_1_chnls, out_channels=lvl_2_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=lvl_2_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block2 = UpConv3DBlock(in_channels=bottleneck_channel, skip_channels=lvl_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=lvl_2_chnls, skip_channels=lvl_1_chnls, out_channels=out_channels,
                                      last=True)

    def forward(self, x):
        # Left path
        out, residual_lvl1 = self.a_block1(x)
        out, residual_lvl2 = self.a_block2(out)
        out, _ = self.bottleNeck(out)

        # Right path
        out = self.s_block2(out, residual_lvl2)
        out = self.s_block1(out, residual_lvl1)
        return out
