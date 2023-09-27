import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from os import path
import sys

sys.path.append("models")


def get_backbone(
    backbone_model_name, num_class, normal_channel, backbone_pretrained_path=None
):
    model_class = importlib.import_module(backbone_model_name)
    backbone_model = model_class.get_model(num_class, normal_channel)
    if backbone_pretrained_path and path.isfile(backbone_pretrained_path):
        checkpoint = torch.load(backbone_pretrained_path)
        backbone_model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded pretrained backbone model")

    return backbone_model


class get_model(nn.Module):
    def __init__(
        self,
        backbone_model_name: str,
        backbone_pretrained_path: str = None,
        backbone_frozen: bool = True,
        backbone_outdims: int = 256,
        num_class: int = 42,
        normal_channel: bool = True,
        n_out_dims: int = 10,
    ):
        super(get_model, self).__init__()
        self.backbone = get_backbone(
            backbone_model_name,
            num_class,
            normal_channel,
            backbone_pretrained_path,
        )
        if backbone_frozen:
            self.backbone = self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(backbone_outdims, backbone_outdims * 4),
            nn.ReLU(),
            nn.Linear(backbone_outdims * 4, backbone_outdims),
        )

        self.fc = nn.Sequential(nn.Linear(backbone_outdims, n_out_dims))

    def forward(self, xyz):
        x = self.backbone(xyz.transpose(2, 1), encode_only=True)
        x = self.mlp(x)
        outs = self.fc(x)
        return outs


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred, target)

        return total_loss
