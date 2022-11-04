# Definition of different models

import torch
import segmentation_models_pytorch as smp


class UNet2DRemake:
    def __init__(self):
        self.encoder = 'resnext101_32x8d'
        self.encoder_weights = 'imagenet'
        self.activation = 'sigmoid'

        self.model = smp.Unet(encoder_name=self.encoder, encoder_weights=self.encoder_weights, classes=1,
                              activation=self.activation, in_channels=1)
        return self.model
