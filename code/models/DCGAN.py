import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, channels, kernel_sizes, strides, paddings, batch_norm, activations):
        """
        channels: array containing channels for each CONV layer, starting from the z_dim until the output image channels
        kernel_sizes: array containing kernel sizes for each CONV layer
        strides: array containing strides for each CONV layer
        paddings: array containing paddings for each CONV layer
        batch_norm: True if batch norm is to be used, otherwise, False
        activations: array containing a hidden activation and an output activation
        """
        super(Generator, self).__init__()
        self.model = self._build(channels, kernel_sizes, strides, paddings, batch_norm, activations)
        self._init_weights()
    
    def _build(self, channels, kernel_sizes, strides, paddings, batch_norm, activations):
        pass

    def _init_weights(self):
        pass

    def forward(self, x):
        x = self.model(x)
        return x