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
        for i in range(1, len(channels)):
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
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, channels, kernel_sizes, strides, paddings, batch_norm, activation):
        """
        channels: array containing channels for each conv layer, starting from the input image dimension until the output dimension
        kernel_sizes: array containing kernel sizes for each conv layer
        strides: array containing strides for each conv layer
        paddings: array containing paddings for each conv layer
        batch_norm: True if batch norm is to be used, otherwise, False
        activation: a single activation function for the hidden layers
        """
        super(Discriminator, self).__init__()
        self.model = self._build(channels, kernel_sizes, strides, paddings, batch_norm, activation)
        self._init_weights()

    def _build(self, channels, kernel_sizes, strides, paddings, batch_norm, activation):
        layers = []
        for i in range(1, len(channels)):
            # add convtranspose layer
            layers += [nn.Conv2d(
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
            # add hidden activation
            if i < len(channels)-1:
                layers += [activation]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Generator G
    Z_DIM = 100
    G = Generator(
        channels=[Z_DIM, 256, 128, 64, 1],
        kernel_sizes=[None, 7, 5, 4, 4],
        strides=[None, 1, 1, 2, 2],
        paddings=[None, 0, 2, 1, 1],
        batch_norm=True,
        activations=[nn.ReLU(), nn.Tanh()]
    )
    print(G)

    # Discriminator D
    D = Discriminator(
        channels=[1, 64, 128, 256, 1],
        kernel_sizes=[None, 4, 4, 5, 7],
        strides=[None, 2, 2, 1, 1],
        paddings=[None, 1, 1, 2, 0],
        batch_norm=False,
        activation=nn.LeakyReLU(0.2),
    )
    print(D)