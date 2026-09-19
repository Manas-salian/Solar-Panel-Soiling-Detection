"""
Model factory built on torchvision's pretrained ImageNet backbones.

Every supported backbone exposes:
    * a feature extractor  (model.features  or  model.layer1..layer4 for ResNet)
    * a classification head (model.classifier  or  model.fc)

We swap the final Linear layer for a NUM_CLASSES-way head and provide helpers
to freeze / unfreeze the backbone and to locate the Grad-CAM target layer.
"""

import torch
import torch.nn as nn
from torchvision.models import get_model

SUPPORTED_BACKBONES = [
    "mobilenet_v2",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "convnext_tiny",
    "resnet50",
]


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {SUPPORTED_BACKBONES}")
    model = get_model(backbone, weights="DEFAULT" if pretrained else None)
    _replace_head(model, num_classes)
    model.backbone_name = backbone
    return model


def _replace_head(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):          # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return
    clf = model.classifier                                                  # MobileNet / EfficientNet / ConvNeXt
    if isinstance(clf, nn.Linear):
        model.classifier = nn.Linear(clf.in_features, num_classes)
        return
    last_idx = max(i for i, m in enumerate(clf) if isinstance(m, nn.Linear))
    clf[last_idx] = nn.Linear(clf[last_idx].in_features, num_classes)


def head_module(model: nn.Module) -> nn.Module:
    return model.fc if hasattr(model, "fc") else model.classifier


def head_parameters(model: nn.Module):
    return list(head_module(model).parameters())


def backbone_parameters(model: nn.Module):
    head_ids = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad = not frozen


def set_train_mode(model: nn.Module, backbone_frozen: bool) -> None:
    """model.train(), but keep BatchNorm running statistics fixed inside a frozen backbone."""
    model.train()
    if backbone_frozen:
        head = head_module(model)
        for m in model.modules():
            if m is not head and not _is_inside(m, head):
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    m.eval()


def _is_inside(module: nn.Module, parent: nn.Module) -> bool:
    return any(module is m for m in parent.modules())


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block: spatial map right before global pooling."""
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4
    raise ValueError("Cannot locate a Grad-CAM target layer for this model.")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


@torch.no_grad()
def measure_latency_ms(model: nn.Module, img_size: int, device, n_iters: int = 50) -> float:
    """Average single-image forward time in milliseconds on `device`."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000.0
