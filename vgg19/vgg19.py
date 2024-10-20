import torch
from torch import nn
import numpy as np



KERNEL_SIZE = (3,3)

class VGG19(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, KERNEL_SIZE, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, KERNEL_SIZE, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, KERNEL_SIZE, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, KERNEL_SIZE, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(49*512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    

model = VGG19()


state_dict = torch.load("models/vgg19-dcbb9e9d.pth")

model.load_state_dict(state_dict, strict=True)
print(state_dict.keys())
print()
print(model.state_dict().keys())
print()

print(model.state_dict()["classifier.0.weight"].shape)


print(state_dict["classifier.0.weight"].shape)