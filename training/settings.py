"""User-tunable settings for the training tools.

Folder names, the Tesseract lookup, dataset limits and the training defaults all
live here, so the tools themselves hold no machine-specific values.
"""

from __future__ import annotations

PROTOCOL = 2
FILE_SET = "2026-09-06-a"

# --- folders and files ---------------------------------------------------
CONFIG_PATH = "config.json"      # slot boxes produced by regions.py
SHOTS_DIR = "shots"              # screenshots to build the dataset from
DATASET_DIR = "dataset"          # patches waiting for a label
CLEAN_DIR = "clean"              # patches that have been labelled
MODEL_PATH = "digits.npz"        # trained weights written by train.py

# --- reading (shared with the app) ---------------------------------------
MODEL_MIN_SCORE = 0.60           # accept a digit only above this probability
MODEL_MIN_MARGIN = 0.20          # ...and only if it beats the runner-up by this

# --- Tesseract -----------------------------------------------------------
# Searched in order after the --tesseract option and the TESSERACT_CMD
# environment variable. Environment variables in the paths are expanded.
TESSERACT_ENV = "TESSERACT_CMD"
TESSERACT_PLACES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe",
    r"%USERPROFILE%\AppData\Local\Tesseract-OCR\tesseract.exe",
)

# --- dataset -------------------------------------------------------------
OTHER_CAP = 40                   # non-digit patches kept per screenshot
DIGIT_LIKE = 0.80                # above this a patch goes to manual review
INCOMPLETE_SCORE = 0.85          # stricter bar while few digits are known

# --- training ------------------------------------------------------------
HIDDEN = 96                      # hidden layer width
EPOCHS = 60                      # passes over the training data
TARGET = 1500                    # samples per class after augmentation
DECAY = 1e-4                     # weight decay
SMOOTHING = 0.05                 # label smoothing
SEED = 0
