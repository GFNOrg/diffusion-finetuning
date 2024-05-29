import torch.nn as nn
import torch
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, input_size, channels, num_classes):
        super(CNN, self).__init__()
        self.input_shape = (channels, input_size, input_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embed = nn.Sequential(self.conv1, self.conv2)

        x = self.embed(torch.zeros((1,) + tuple(self.input_shape)))
        input_dim = x.view(x.size(0), -1).shape[-1]
        self.out = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class ResNet(nn.Module):

    name = "ResNet"

    def __init__(self, input_size, channels, num_classes, depth=18, finetune=False):
        super(ResNet, self).__init__()
        self.input_shape = (channels, input_size, input_size)
        self.depth = depth
        self.num_classes = num_classes

        # create the embedding (sub) model
        self.resnet = self._init_resnet(finetune)

        if finetune:
            for name, param in self.resnet.named_parameters():
                param.requires_grad = False

        # we dynamically create readout layer dimensions
        x = self.resnet(torch.zeros((2,) + tuple(self.input_shape)))

        # make sure to set the input dimension, so the paent class knows how to build the models
        input_dim = x.view(x.size(0), -1).shape[-1]

        # creates and initializes all models and associated optimizers within ensemble
        self.out_layer = self._create_readout_layer(input_dim, num_classes)

    def _init_resnet(self, finetune=False):

        # Dynamically select the ResNet architecture
        resnet_class = getattr(models, f'resnet{self.depth}')

        # Initialize a pre-trained model
        resnet_model = resnet_class(weights=models.ResNet18_Weights.DEFAULT if finetune else None)
        # resnet_model = resnet_class(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer (for different input channels)
        if self.input_shape[0] != 3:
            resnet_model.conv1 = torch.nn.Conv2d(self.input_shape[0], 64,
                                                 kernel_size=(7, 7),
                                                 stride=(2, 2),
                                                 padding=(0, 3),
                                                 bias=False)

        modules = list(resnet_model.children())[:-1]  # Exclude the last fc layer
        resnet_model = nn.Sequential(*modules)

        return resnet_model

    def _create_readout_layer(self, input_dimension, output_dimension):
        return nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dimension, output_dimension))

    def forward(self, x):
        x = self.resnet(x)
        return self.out_layer(x)
