# setup_project.py
# Esegui da dentro la cartella BART:
#   python setup_project.py

import os


def create_project():

    # ── struttura cartelle ────────────────────────────────────────
    dirs = [
        "configs",
        "data",
        "model",
        "training",
        "checkpoints",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()

    print("Cartelle create.")

    # ── file vuoti con commento iniziale ──────────────────────────
    files = {
        "configs/bart_base.py":     "# Configurazione per BART-base\n",
        "data/dataset.py":          "# Caricamento e tokenizzazione dataset\n",
        "data/translation.py":      "# Dataset per traduzione (opus_books)\n",
        "model/model.py":           "# Caricamento modello\n",
        "training/metrics.py":      "# Metriche: ROUGE, BLEU, BERTScore\n",
        "training/trainer.py":      "# Seq2SeqTrainer e TrainingArguments\n",
        "train.py":                 "# Entry point — avvia il training\n",
        "predict.py":               "# Inferenza su nuove frasi\n",
        "plot.py":                  "# Visualizzazione loss e metriche\n",
        "requirements.txt": (
            "torch\n"
            "transformers\n"
            "datasets\n"
            "evaluate\n"
            "accelerate>=1.1.0\n"
            "rouge-score\n"
            "nltk\n"
            "absl-py\n"
            "bert-score\n"
            "matplotlib\n"
            "sacrebleu\n"
        ),
        ".gitignore": (
            "checkpoints/\n"
            "__pycache__/\n"
            "*.pyc\n"
            ".env\n"
            ".venv/\n"
            "*.pt\n"
            "*.safetensors\n"
        ),
        "README.md": (
            "# BART — Encoder-Decoder\n\n"
            "## Setup\n\n"
            "```bash\n"
            "pip install -r requirements.txt\n"
            "```\n\n"
            "## Training\n\n"
            "```bash\n"
            "python train.py --config configs/bart_base.py\n"
            "```\n\n"
            "## Plot risultati\n\n"
            "```bash\n"
            "python plot.py --config configs/bart_base.py\n"
            "```\n"
        ),
    }

    for path, content in files.items():
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(content)

    print("File creati.")

    # ── riepilogo ─────────────────────────────────────────────────
    print("\nStruttura progetto BART:\n")
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs
            if d not in ["__pycache__", "checkpoints", ".git"]
        ]
        level  = root.count(os.sep)
        indent = "    " * level
        folder = os.path.basename(root)
        if folder == ".":
            folder = "BART/"
        else:
            print(f"{indent}{folder}/")
            indent = "    " * (level + 1)
        for f in sorted(files):
            if not f.endswith(".pyc"):
                print(f"{indent}{f}")

    print("\nProssimo passo:")
    print("  pip install -r requirements.txt")
    print("  python train.py --config configs/bart_base.py")


if __name__ == "__main__":
    create_project()