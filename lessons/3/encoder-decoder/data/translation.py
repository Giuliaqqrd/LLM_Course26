# data/translation.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL, MAX_INPUT_LENGTH, MAX_TRAIN_SAMPLES, MAX_EVAL_SAMPLES, SEED, get_task_config


def load_and_tokenize():

    task_cfg = get_task_config()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    dataset = load_dataset(task_cfg["name"], task_cfg["config"])

    # opus_books non ha validation split, lo creiamo dividendo il train
    # prima divisione: 80% train, 20% temp
    split1 = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
    # seconda divisione: temp -> 50% validation, 50% test
    split2 = split1["test"].train_test_split(test_size=0.5, seed=SEED)

    dataset["train"]      = split1["train"]
    dataset["validation"] = split2["train"]
    dataset["test"]       = split2["test"]
    print(f"train: {len(dataset['train'])} — validation: {len(dataset['validation'])}")

    # taglia prima del map
    if MAX_TRAIN_SAMPLES:
        dataset["train"]      = dataset["train"].select(range(MAX_TRAIN_SAMPLES))
    if MAX_EVAL_SAMPLES:
        dataset["validation"] = dataset["validation"].select(range(MAX_EVAL_SAMPLES))

    def tokenize_function(examples):
        # struttura annidata: examples["translation"] = [{"en": "...", "it": "..."}, ...]
        inputs  = [task_cfg["prefix"] + t[task_cfg["input_key"]]
                   for t in examples[task_cfg["input_col"]]]
        targets = [t[task_cfg["target_key"]]
                   for t in examples[task_cfg["target_col"]]]

        model_inputs = tokenizer(
            inputs,
            text_target=targets,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
            padding=False,
        )

        model_inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label]
            for label in model_inputs["labels"]
        ]

        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    return tokenized_dataset, tokenizer


if __name__ == "__main__":
    tokenized_dataset, tokenizer = load_and_tokenize()
    esempio = tokenized_dataset["train"][0]

    print(f"input_ids length: {len(esempio['input_ids'])}")
    print(f"labels length:    {len(esempio['labels'])}")
    print(f"Input:  {tokenizer.decode(esempio['input_ids'][:50])}")

    labels_reali = [t for t in esempio["labels"] if t != -100]
    print(f"Target: {tokenizer.decode(labels_reali[:50])}")