"""
Predict one or more images from the command line.

    python scripts/predict.py path/to/panel.jpg [more.jpg ...] [--model outputs/final_model.pt]
                              [--no-tta] [--cam out_dir] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FINAL_MODEL_PATH  # noqa: E402
from src.inference import Predictor, load_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Classify solar panel images.")
    ap.add_argument("images", nargs="+", help="Image files")
    ap.add_argument("--model", default=str(FINAL_MODEL_PATH))
    ap.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip test-time augmentation")
    ap.add_argument("--cam", metavar="DIR", help="Also write a Grad-CAM overlay per image into DIR")
    ap.add_argument("--json", action="store_true", help="Emit one JSON object per line instead of text")
    args = ap.parse_args()

    predictor = Predictor(args.model)
    if args.cam:
        Path(args.cam).mkdir(parents=True, exist_ok=True)

    for path in args.images:
        img = load_image(path)
        result = predictor.predict(img, tta=not args.no_tta)
        summary = predictor.summarize(result["probs"])
        if args.cam:
            out = Path(args.cam) / f"{Path(path).stem}_cam.jpg"
            predictor.explain(img).save(out, quality=90)
            summary["gradcam"] = str(out)

        if args.json:
            print(json.dumps({"file": path, **summary, "latency_ms": round(result["latency_ms"], 1)}))
            continue

        print(f"\n{path}")
        print(f"  {summary['label']}  ({summary['confidence']:.1%})  severity: {summary['severity']}")
        print(f"  action: {summary['action']}")
        if summary["needs_review"]:
            print("  ! needs manual review: " + " ".join(summary["review_reasons"]))
        print("  all classes: " + ", ".join(f"{c} {p:.1%}" for c, p in summary["ranked"]))
        print(f"  latency: {result['latency_ms']:.1f} ms on {predictor.device}")


if __name__ == "__main__":
    main()
