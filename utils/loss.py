import torch
import torch.nn as nn

from torchvision.models import vgg19


class ContentLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = vgg19(pretrained=True).features
        for param in self.features:
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, out: torch.Tensor, target: torch.Tensor):
        return self.mse_loss(self.features(out), self.features(target))
