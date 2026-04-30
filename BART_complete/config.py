# da copiare nel progetto
import os
from datetime import datetime

TASK      = "summarization"
MODEL = "facebook/bart-base"

DATASET_CONFIG = {
    "summarization": {
        "name":       "cnn_dailymail",
        "config":     "3.0.0",
        "input_col":  "article",
        "target_col": "highlights",
        "prefix":     "",
        "lang":       "en",
    },
    "translation": {
        "name":       "opus_books",
        "config":     "en-it",
        "input_col":  "translation",
        "input_key":  "en",
        "target_col": "translation",
        "target_key": "it",
        "prefix":     "",
        "lang":       "it",
    },
    "simplification": {
        "name":       "wiki_auto",
        "config":     "auto_acl",
        "input_col":  "normal_sentence",
        "target_col": "simple_sentence",
        "prefix":     "",
        "lang":       "en",
    },
}

MAX_INPUT_LENGTH  = 1024
MAX_TARGET_LENGTH = 128
NUM_EPOCHS        = 3
TRAIN_BATCH_SIZE  = 4
EVAL_BATCH_SIZE   = 4
LEARNING_RATE     = 3e-5
EVAL_STRATEGY     = "epoch"
SAVE_STRATEGY     = "epoch"
SAVE_TOTAL_LIMIT  = 2
LOGGING_STEPS     = 100
MAX_TRAIN_SAMPLES = 1000
MAX_EVAL_SAMPLES  = 200
SEED              = 42

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    "checkpoints",
    f"{TASK}_{MODEL}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}_{timestamp}"
)

def get_task_config():
    if TASK not in DATASET_CONFIG:
        raise ValueError(f"Task '{TASK}' non supportato.")
    return DATASET_CONFIG[TASK]