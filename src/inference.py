"""
Single-image inference shared by the Streamlit app and the CLI.

    predictor = Predictor("outputs/final_model.pt")
    result = predictor.predict(pil_image)          # probabilities + timing
    summary = predictor.summarize(result["probs"]) # label, action, confidence flags
    heatmap = predictor.explain(pil_image)         # Grad-CAM overlay (PIL)
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import CLASS_INFO, CLOSE_CALL_MARGIN, LOW_CONFIDENCE_THRESHOLD
from src.data import build_transforms
from src.gradcam import GradCAM, overlay_cam
from src.model import build_model, gradcam_target_layer

SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}


def load_image(source) -> Image.Image:
    """Accept a path, file-like object, or PIL image. Returns RGB with EXIF orientation applied."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


class Predictor:
    def __init__(self, checkpoint_path, device: str | None = None):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No model at {checkpoint_path}. Train one with `python main.py`.")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = ckpt["backbone"]
        self.class_names = list(ckpt["class_names"])
        self.img_size = int(ckpt["img_size"])
        self.meta = {k: v for k, v in ckpt.items() if k != "state_dict"}

        self.model = build_model(self.backbone, len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.transform = build_transforms(self.img_size, train=False)
        self._cam = None

    # ------------------------------------------------------------------ #
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, tta: bool = True) -> dict:
        """Softmax probabilities; with `tta` the horizontal flip is averaged in."""
        x = self.preprocess(image)
        if tta:
            x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
        t0 = time.perf_counter()
        probs = torch.softmax(self.model(x).float(), dim=1).mean(dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = probs.cpu().numpy()
        return {"probs": probs, "top_idx": int(probs.argmax()), "latency_ms": latency_ms, "tta": tta}

    def explain(self, image: Image.Image, class_idx: int | None = None, alpha: float = 0.45) -> Image.Image:
        """Grad-CAM heat-map for `class_idx` (default: predicted class), blended onto the image."""
        if self._cam is None:
            self._cam = GradCAM(self.model, gradcam_target_layer(self.model))
        cam, _ = self._cam(self.preprocess(image), class_idx)
        return overlay_cam(image, cam, alpha=alpha)

    # ------------------------------------------------------------------ #
    def summarize(self, probs: np.ndarray,
                  low_conf_thr: float = LOW_CONFIDENCE_THRESHOLD,
                  margin_thr: float = CLOSE_CALL_MARGIN) -> dict:
        order = np.argsort(probs)[::-1]
        top, second = int(order[0]), int(order[1])
        cls = self.class_names[top]
        info = CLASS_INFO[cls]
        confidence = float(probs[top])
        margin = float(probs[top] - probs[second])

        flags = []
        if confidence < low_conf_thr:
            flags.append(f"Top-1 probability {confidence:.0%} is below the {low_conf_thr:.0%} confidence threshold.")
        if margin < margin_thr:
            flags.append(f"Close call: only {margin:.0%} separates it from "
                         f"'{CLASS_INFO[self.class_names[second]]['label']}'.")

        return {
            "class": cls,
            "label": info["label"],
            "severity": info["severity"],
            "severity_rank": SEVERITY_RANK[info["severity"]],
            "action": info["action"],
            "confidence": confidence,
            "second_class": self.class_names[second],
            "second_label": CLASS_INFO[self.class_names[second]]["label"],
            "second_confidence": float(probs[second]),
            "margin": margin,
            "needs_review": bool(flags),
            "review_reasons": flags,
            "ranked": [(self.class_names[i], float(probs[i])) for i in order],
        }
