# plot.py
import os
import json
import matplotlib.pyplot as plt
from config import MODEL_NAME, TASK


def find_latest_checkpoint(base_dir: str = "checkpoints"):
    if not os.path.exists(base_dir):
        print(f"Cartella '{base_dir}' non trovata.")
        return None

    stati = []
    for root, dirs, files in os.walk(base_dir):
        if "trainer_state.json" in files:
            path = os.path.join(root, "trainer_state.json")
            stati.append((os.path.getmtime(path), path))

    if not stati:
        return None

    stati.sort(reverse=True)
    return stati[0][1]


def plot_training():

    log_path = find_latest_checkpoint()

    if not log_path:
        print("Nessun checkpoint trovato in checkpoints/")
        print("Esegui prima: python train.py")
        return

    print(f"Leggo log da: {log_path}")

    with open(log_path) as f:
        state = json.load(f)

    log_history = state["log_history"]

    train_logs = [l for l in log_history if "loss" in l and "eval_loss" not in l]
    eval_logs  = [l for l in log_history if "eval_loss" in l]

    if not train_logs:
        print("Nessun log di training trovato.")
        return

    train_steps = [l["step"] for l in train_logs]
    train_loss  = [l["loss"] for l in train_logs]

    eval_epochs = [l["epoch"]     for l in eval_logs]
    eval_loss   = [l["eval_loss"] for l in eval_logs]

    rouge1    = [l.get("eval_rouge1")       for l in eval_logs]
    rouge2    = [l.get("eval_rouge2")       for l in eval_logs]
    rougeL    = [l.get("eval_rougeL")       for l in eval_logs]
    bertscore = [l.get("eval_bertscore_f1") for l in eval_logs]

    has_rouge     = any(v is not None for v in rouge1)
    has_bertscore = any(v is not None for v in bertscore)

    # calcola numero di grafici dinamicamente
    n_plots = 2
    if has_rouge:     n_plots += 1
    if has_bertscore: n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    fig.suptitle(f"{MODEL_NAME} — {TASK}", fontsize=13)

    # -- training loss per step ────────────────────────────────────────
    axes[0].plot(train_steps, train_loss, color="steelblue", linewidth=1)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # -- validation loss per epoca ─────────────────────────────────────
    if eval_logs:
        axes[1].plot(eval_epochs, eval_loss, marker="o", color="coral")
        axes[1].set_title("Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)

    # -- ROUGE per epoca ───────────────────────────────────────────────
    if has_rouge:
        axes[2].plot(eval_epochs, rouge1, marker="o", label="ROUGE-1")
        axes[2].plot(eval_epochs, rouge2, marker="o", label="ROUGE-2")
        axes[2].plot(eval_epochs, rougeL, marker="o", label="ROUGE-L")
        axes[2].set_title("ROUGE scores")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Score")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # -- BERTScore per epoca ───────────────────────────────────────────
    if has_bertscore:
        ax_idx = 3 if has_rouge else 2
        axes[ax_idx].plot(eval_epochs, bertscore, marker="o", color="mediumpurple")
        axes[ax_idx].set_title("BERTScore F1")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Score")
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join("checkpoints", "training_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato in: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_training()