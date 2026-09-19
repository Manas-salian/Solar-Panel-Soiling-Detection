"""
Minimal Grad-CAM (Selvaraju et al., 2017) for explaining a single prediction.

Hooks the output of the last convolutional block, back-propagates the score of
the chosen class, and weights the activation maps by their mean gradient.
"""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self._acts = output
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self._grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """x: [1, 3, H, W] on the model's device. Returns (cam [H, W] in [0, 1], class_idx)."""
        self.model.eval()
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)            # [1, C, 1, 1]
        cam = (weights * self._acts).sum(dim=1, keepdim=True).relu()     # [1, 1, h, w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().float().cpu().numpy(), class_idx

    def remove(self):
        self._handle.remove()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, cmap: str = "inferno") -> Image.Image:
    """Blend a [0, 1] heat-map onto the original image (any size)."""
    heat = colormaps[cmap](cam)[..., :3]                                 # [h, w, 3] in 0..1
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heat_img, alpha)
