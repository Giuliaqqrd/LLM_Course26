# model/model.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import AutoModelForSeq2SeqLM
from config import MODEL


def load_model():

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

    print(f"Modello: {MODEL}")
    print(f"Parametri totali:     {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parametri allenabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


if __name__ == "__main__":
    model = load_model()
    print(f"\nArchitettura:\n{model}")