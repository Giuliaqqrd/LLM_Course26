# training/trainer.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoTokenizer
from config import (
    MODEL,
    OUTPUT_DIR,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    EVAL_STRATEGY,
    SAVE_STRATEGY,
    SAVE_TOTAL_LIMIT,
    LOGGING_STEPS,
    SEED,
)


def build_training_args():

    return Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        predict_with_generate=True,
        # ── NUOVO rispetto a DistilBERT ───────────────────────────────
        # dice al Trainer di usare model.generate() durante la valutazione
        # invece di prendere l'argmax dei logit
        #
        # senza questo:
        #   Trainer prende argmax(logits) → token più probabile per posizione
        #   ignora che ogni token dipende da quelli precedenti
        #   ROUGE sarebbe calcolato su sequenze sbagliate
        #
        # con questo:
        #   Trainer chiama model.generate() → generazione autoregressiva
        #   ogni token viene generato condizionatamente ai precedenti
        #   ROUGE viene calcolato su sequenze realistiche
    )


def build_trainer(model, tokenized_dataset, tokenizer, compute_metrics):

    training_args = build_training_args()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # ── NUOVO rispetto a DistilBERT ───────────────────────────────
        # DataCollatorForSeq2Seq invece di DataCollatorWithPadding
        #
        # DataCollatorWithPadding:  padda solo input_ids e attention_mask
        # DataCollatorForSeq2Seq:   padda anche le labels
        #                           e le padda con -100 automaticamente
        #                           così la loss le ignora
        #
        # serve il modello (model=model) perché T5 ha bisogno di sapere
        # il pad_token_id per paddare le labels correttamente
        padding=True,
        label_pad_token_id=-100,
        # conferma esplicitamente che il padding delle labels usa -100
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    return trainer


if __name__ == "__main__":
    args = build_training_args()
    print(f"Output dir:           {args.output_dir}")
    print(f"Epoche:               {args.num_train_epochs}")
    print(f"Batch size:           {args.per_device_train_batch_size}")
    print(f"Learning rate:        {args.learning_rate}")
    print(f"predict_with_generate:{args.predict_with_generate}")