import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, channels, kernel_sizes, strides, paddings, batch_norm, activations):
        """
        channels: array containing channels for each convtranspose layer, starting from the input dimension until the output image channels
        kernel_sizes: array containing kernel sizes for each convtranspose layer
        strides: array containing strides for each convtranspose layer
        paddings: array containing paddings for each convtranspose layer
        batch_norm: True if batch norm is to be used, otherwise, False
        activations: array containing a hidden activation and an output activation
        """
        super(Generator, self).__init__()
        self.model = self._build(channels, kernel_sizes, strides, paddings, batch_norm, activations)
        self._init_weights()
    
    def _build(self, channels, kernel_sizes, strides, paddings, batch_norm, activations):
        layers = []
        for i in range(1, channels):
            # add convtranspose layer
            layers += [nn.ConvTranspose2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                bias=False if batch_norm else True
            )]
            # add batchnorm
            if batch_norm and i < len(channels)-1:
                layers += [nn.BatchNorm2d(channels[i])]
            # add activation
            layers += [activations[0] if i < len(channels)-1 else activations[1]]
        return nn.Sequential(*layers)

    def _init_weights(self):
        pass

    def forward(self, x):
        x = self.model(x)
        return x