# train.py
import os
import sys
import random
import numpy as np
import torch

from config import OUTPUT_DIR, SEED, TASK
from model.model import load_model
from training.metrics import compute_metrics
from training.trainer import build_trainer

if TASK == "summarization":
    from data.summarization import load_and_tokenize
pass # se TASK è "translation", importa load_and_tokenize da data/translation.py



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(SEED)
    print(f"Task:  {TASK}")
    print(f"Seed:  {SEED}")

    # -- 1. dati --------------------------------------------------
    print("\n--- Caricamento dataset ---")
    tokenized_dataset, tokenizer = load_and_tokenize()

    # -- 2. modello -----------------------------------------------
    print("\n--- Caricamento modello ---")
    model = load_model()

    # -- 3. trainer -----------------------------------------------
    print("\n--- Configurazione trainer ---")
    trainer = build_trainer(
        model=model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -- 4. training ----------------------------------------------
    print("\n--- Avvio training ---")
    trainer.train()

    # -- 5. valutazione -------------------------------------------
    print("\n--- Valutazione finale ---")
    risultati = trainer.evaluate()
    print(f"ROUGE-1: {risultati['eval_rouge1']}")
    print(f"ROUGE-2: {risultati['eval_rouge2']}")
    print(f"ROUGE-L: {risultati['eval_rougeL']}")
    print(f"BERTScore F1: {risultati['eval_bertscore_f1']}")

    # -- 6. salvataggio -------------------------------------------
    print(f"\n--- Salvataggio ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modello salvato in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()