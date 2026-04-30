# predict.py
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")    # ← Mac Apple Silicon
else:
    device = torch.device("cpu")

print(f"Device: {device}")
def find_best_model(base_dir: str = "checkpoints"):
    """Trova il modello più recente in checkpoints/."""
    if not os.path.exists(base_dir):
        return None

    checkpoints = []
    for root, dirs, files in os.walk(base_dir):
        if "trainer_state.json" in files:
            checkpoints.append((os.path.getmtime(root), root))

    if not checkpoints:
        return None

    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def load_model(model_dir: str):
    print(f"Carico modello da: {model_dir}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return tokenizer, model


def generate(text: str, tokenizer, model):
    prefix     = "summarize: " if "t5" in MODEL else ""
    input_text = prefix + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    # sposta gli input sullo stesso device del modello
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_new_tokens=128,
             max_length=None,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():

    model_dir = find_best_model()
    if not model_dir:
        print("Nessun modello trovato in checkpoints/")
        print("Esegui prima: python train.py")
        sys.exit(1)

    tokenizer, model = load_model(model_dir)

    print(f"Modello: {MODEL}")
    print("Scrivi un articolo da riassumere. Digita 'exit' per uscire.\n")
    print("-" * 60)

    while True:
        print("\nArticolo:")
        testo = input("> ").strip()

        if testo.lower() == "exit":
            print("Arrivederci!")
            break

        if not testo:
            print("Testo vuoto — riprova.")
            continue

        print("\nRiassunto:")
        riassunto = generate(testo, tokenizer, model)
        print(riassunto)
        print("-" * 60)


if __name__ == "__main__":
    main()