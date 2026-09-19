"""
Shared configuration for the Solar Panel Fault & Soiling Detection project.
Everything that more than one module needs lives here.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RAW_DATA_DIR = PROJECT_ROOT / "Faulty_solar_panel"      # Kaggle download, untouched
DATA_DIR = PROJECT_ROOT / "data"                         # produced by scripts/prepare_dataset.py
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FINAL_MODEL_PATH = OUTPUT_DIR / "final_model.pt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "comparison.json"
DATASET_REPORT_PATH = OUTPUT_DIR / "dataset_report.json"

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# Alphabetical order == torchvision.datasets.ImageFolder order. Do not reorder.
CLASS_NAMES = [
    "bird_drop",
    "clean",
    "dusty",
    "electrical_damage",
    "physical_damage",
    "snow_covered",
]
NUM_CLASSES = len(CLASS_NAMES)

# Human-facing metadata used by the UI and CLI.
# severity: none < low < moderate < high < critical
CLASS_INFO = {
    "bird_drop": {
        "label": "Bird Droppings",
        "severity": "moderate",
        "action": "Spot-clean the affected cells soon. Droppings block light locally and "
                  "can create hot-spots that permanently damage the cell.",
    },
    "clean": {
        "label": "Clean",
        "severity": "none",
        "action": "No action needed. Panel is operating normally.",
    },
    "dusty": {
        "label": "Dusty / Soiled",
        "severity": "moderate",
        "action": "Schedule a wash. Uniform soiling typically cuts output by 5-25% "
                  "depending on thickness.",
    },
    "electrical_damage": {
        "label": "Electrical Damage",
        "severity": "critical",
        "action": "Isolate the string and dispatch a technician. Burn marks, hot-spots or "
                  "arc faults are a fire risk and will not self-resolve.",
    },
    "physical_damage": {
        "label": "Physical Damage",
        "severity": "high",
        "action": "Inspect on site. Cracked or shattered glass lets in moisture; the module "
                  "usually needs replacement.",
    },
    "snow_covered": {
        "label": "Snow Covered",
        "severity": "low",
        "action": "Usually self-clears as the panel warms. Clear manually only if the outage "
                  "is prolonged; never use sharp tools on the glass.",
    },
}

_CLASS_ALIASES = {
    "bird-drop": "bird_drop", "bird_drop": "bird_drop", "birddrop": "bird_drop",
    "clean": "clean",
    "dusty": "dusty", "dust": "dusty",
    "electrical-damage": "electrical_damage", "electrical_damage": "electrical_damage",
    "physical-damage": "physical_damage", "physical_damage": "physical_damage",
    "snow-covered": "snow_covered", "snow_covered": "snow_covered", "snow": "snow_covered",
}


def normalize_class_name(folder_name: str) -> str:
    """Map a raw folder name such as 'Bird-drop' to the canonical 'bird_drop'."""
    key = folder_name.strip().lower().replace(" ", "_")
    return _CLASS_ALIASES.get(key, key.replace("-", "_"))


# --------------------------------------------------------------------------- #
# Training defaults (overridable from main.py CLI)
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
NUM_WORKERS = 6

DEFAULT_BACKBONES = ["mobilenet_v3_large", "efficientnet_v2_s", "convnext_tiny"]
DEFAULT_EPOCHS = 30          # full fine-tune stage (early stopping usually ends it sooner)
DEFAULT_WARMUP_EPOCHS = 3    # head-only stage with the backbone frozen
EARLY_STOPPING_PATIENCE = 8
LABEL_SMOOTHING = 0.1
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
WEIGHT_DECAY = 0.02

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
LOW_CONFIDENCE_THRESHOLD = 0.60   # top-1 probability below this => flag for manual review
CLOSE_CALL_MARGIN = 0.15          # top-1 minus top-2 below this => also flag
