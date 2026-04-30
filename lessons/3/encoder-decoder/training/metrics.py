# training/metrics.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import evaluate
from transformers import AutoTokenizer
from config import MODEL

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # sostituisce valori negativi con pad_token_id prima di decodificare
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # sostituisce -100 con pad_token_id prima di decodificare
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # -- ROUGE ───────────────────────────────────────────────────────

    rouge_results = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # -- BERTScore ─────────────────────────────────────────────────────
    bert_results = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="it"
        # cambia in base alla lingua del dataset
    )
    # bertscore restituisce una lista di valori: uno per ogni esempio
    # prendiamo la media su tutti gli esempi del batch
    bert_f1 = float(np.mean(bert_results["f1"]))

    return {
        "rouge1": round(rouge_results["rouge1"], 4),
        "rouge2": round(rouge_results["rouge2"], 4),
        "rougeL": round(rouge_results["rougeL"], 4),
        "bertscore_f1": round(bert_f1, 4),
    }


if __name__ == "__main__":
    preds  = ["Harry Potter gets fortune as he turns 18"]
    labels = ["Daniel Radcliffe gets £20M fortune at 18"]

    pred_ids  = tokenizer(preds,  return_tensors="np")["input_ids"]
    label_ids = tokenizer(labels, return_tensors="np")["input_ids"]

    result = compute_metrics((pred_ids, label_ids))
    print(f"ROUGE-1: {result['rouge1']}")
    print(f"ROUGE-2: {result['rouge2']}")
    print(f"ROUGE-L: {result['rougeL']}")
    print(f"BERTScore F1: {result['bertscore_f1']}")