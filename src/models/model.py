import torch
import torch.nn as nn
from torchvision import models

class VGGFaceEmbedder(nn.Module):
    def __init__(self):
        super(VGGFaceEmbedder, self).__init__()
        self.vggface = models.vgg16(pretrained=True)
        self.vggface.classifier = nn.Sequential(*list(self.vggface.classifier.children())[:-1])  # remove last layer

    def forward(self, x):
        return self.vggface(x)


class VGGFaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VGGFaceClassifier, self).__init__()
        self.fc = nn.Linear(4096, num_classes)  # Adjust input size if needed

    def forward(self, x):
        return self.fc(x)
