# config.py
import os
from datetime import datetime

# ── task ──────────────────────────────────────────────────────────
TASK = "summarization"
# cambia questo per passare da un task all'altro:
#   "summarization": riassunto di articoli (cnn_dailymail)
#   "translation": traduzione inglese→italiano (opus_books)
#   "simplification" semplificazione testo (wiki_auto)

# ── modello ───────────────────────────────────────────────────────
MODEL = "t5-small"
# T5-small: 60M parametri, gira su CPU in tempi ragionevoli
# T5-base:  220M parametri, più lento ma risultati migliori
# la stessa architettura funziona per tutti e tre i task 
# cambia solo come prepariamo i dati e come valutiamo

# ── dataset per task ──────────────────────────────────────────────
DATASET_CONFIG = {
    "summarization": {
        "name":         "cnn_dailymail",
        "config":       "3.0.0",
        "input_col":    "article",
        "target_col":   "highlights",
        "prefix":       "summarize: ",
        # T5 usa un prefisso testuale per capire il task 
        # "summarize: " dice al modello cosa deve fare
    },
    "translation": {
        "name":       "opus_books",
        "config":     "en-it",
        "input_col":  "translation",
        "input_key":  "en",
        "target_col": "translation",
        "target_key": "it",
        "prefix":     "translate English to Italian: ",
},
    "simplification": {
        "name":         "wiki_auto",
        "config":       "auto_acl",
        "input_col":    "normal_sentence",
        "target_col":   "simple_sentence",
        "prefix":       "simplify: ",
    },
}

# ── iperparametri ─────────────────────────────────────────────────
MAX_INPUT_LENGTH  = 512     # lunghezza massima della sequenza in input
MAX_TARGET_LENGTH = 128     # lunghezza massima della sequenza in output
                            # più corta dell'input, i riassunti sono brevi

NUM_EPOCHS               = 3
TRAIN_BATCH_SIZE         = 8
                          # più piccolo rispetto a DistilBERT —
                          # T5 è encoder-decoder, usa più memoria
EVAL_BATCH_SIZE          = 8
LEARNING_RATE            = 5e-5
                          # leggermente più alto di DistilBERT —
                          # T5-small richiede un learning rate più aggressivo

# ── valutazione ───────────────────────────────────────────────────
EVAL_STRATEGY    = "epoch"
SAVE_STRATEGY    = "epoch"
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS    = 100

# ── campionamento per la demo ─────────────────────────────────────
MAX_TRAIN_SAMPLES = 1000    # None per usare tutto il dataset
MAX_EVAL_SAMPLES  = 200

# ── output ────────────────────────────────────────────────────────
SEED = 42

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(
    "checkpoints",
    f"{TASK}_{MODEL}_lr{LEARNING_RATE}_ep{NUM_EPOCHS}_{timestamp}"
)
# esempio:
# checkpoints/summarization_t5-small_lr5e-05_ep3_20250426_143512

# ── helper ────────────────────────────────────────────────────────
def get_task_config():
    """
    Restituisce la configurazione del task corrente.
    Usata da dataset.py per sapere quale dataset caricare
    e quali colonne usare.
    """
    if TASK not in DATASET_CONFIG:
        raise ValueError(f"Task '{TASK}' non supportato. Scegli tra: {list(DATASET_CONFIG.keys())}")
    return DATASET_CONFIG[TASK]