"""
Shared configuration constants for the Solar Panel Soiling Detection project.
"""

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["clean", "dusty"]  # alphabetical order, matches Keras loader
PREDICTION_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.65

DATA_DIR = "data"
OUTPUT_DIR = "outputs"

DEFAULT_EPOCHS = 15
DEFAULT_FINE_TUNE_EPOCHS = 5
