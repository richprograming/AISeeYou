import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


class Encoder(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        # ViTs need dynamic_img_size to accept 2:1 panoramas
        kwargs = {"dynamic_img_size": True} if name.startswith("vit") else {}
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
        cfg = self.backbone.pretrained_cfg
        self.mean, self.std = cfg["mean"], cfg["std"]

    def forward(self, x):
        return F.normalize(self.backbone(x), dim=-1)

    def transform(self, size):
        return transforms.Compose([
            transforms.Resize(tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])


@torch.no_grad()
def embed(model, loader, device, log_every=20):
    out = []
    for i, x in enumerate(loader):
        out.append(model(x.to(device)).cpu())
        if log_every and (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(loader)} batches")
    return torch.cat(out)
