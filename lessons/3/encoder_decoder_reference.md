# Encoder-Decoder with HuggingFace — Reference Guide

> Guida di riferimento completa per il corso LLM & AI Generativa  
> Lezione 2 e 3 — Encoder-only e Encoder-decoder

---

## Indice

1. [Architettura Encoder-Decoder](#1-architettura-encoder-decoder)
2. [Differenze con Encoder-Only](#2-differenze-con-encoder-only)
3. [HuggingFace Transformers — classi usate](#3-huggingface-transformers--classi-usate)
4. [Il dataset e la tokenizzazione](#4-il-dataset-e-la-tokenizzazione)
5. [Training — componenti chiave](#5-training--componenti-chiave)
6. [Metriche — ROUGE](#6-metriche--rouge)
7. [T5 — text-to-text](#7-t5--text-to-text)
8. [Struttura del progetto](#8-struttura-del-progetto)
9. [Domande frequenti](#9-domande-frequenti)

---

## 1. Architettura Encoder-Decoder

### Panoramica

Un modello encoder-decoder è composto da due parti distinte che collaborano:

- **Encoder** — legge l'intera sequenza in input simultaneamente usando self-attention bidirezionale. Produce una rappresentazione contestuale di ogni token.
- **Decoder** — genera l'output token per token, condizionato alla rappresentazione prodotta dall'encoder.

```
Input sequence
      ↓
Embedding + Positional Encoding
      ↓
Transformer Encoder × N
(bidirectional self-attention + feedforward)
      ↓
Encoded representation
      ↓
Transformer Decoder × N
(masked self-attention + cross-attention + feedforward)
      ↓
Output token (one at a time)
```

---

### I tre meccanismi del decoder

**1. Masked self-attention**

Il decoder può guardare solo i token che ha già generato, non quelli futuri. Una maschera triangolare blocca l'accesso ai token successivi:

```
Generando "come":
  può vedere:    [BOS] ciao
  non può vedere: stai [EOS]
```

Perché? A tempo di inferenza i token futuri non esistono ancora. La maschera forza il modello a comportarsi allo stesso modo durante il training.

**2. Cross-attention**

L'unico punto dove encoder e decoder si incontrano. Il decoder chiede: *"quale parte dell'input è più rilevante per quello che sto generando adesso?"*

```
Query  -> viene dal decoder  (token corrente)
Keys   -> vengono dall'encoder (intera sequenza input)
Values -> vengono dall'encoder (intera sequenza input)
```

Matematicamente:

```
scores     = Q · Kᵀ / √dk
weights    = softmax(scores)
output     = weights · V
```

**3. Feedforward network**

Identico all'encoder — trasforma ogni vettore indipendentemente dopo l'attention.

---

### Training — Teacher Forcing

Durante il training, encoder e decoder ricevono i loro input **in parallelo**:

```
Encoder riceve:  "hello how are you"          ← input originale
Decoder riceve:  [BOS] ciao come stai         ← ground truth shifted right
```

Il decoder non usa le proprie predizioni come input — riceve sempre la sequenza corretta dal dataset. Questo si chiama **teacher forcing**.

**Shifted right** — il target viene spostato di una posizione a destra:

```
Target:         ciao  come  stai  [EOS]
Decoder input:  [BOS] ciao  come  stai
```

Ad ogni posizione il decoder vede i token precedenti corretti e deve predire il token successivo.

**Perché teacher forcing?**

Senza di esso, un errore al token 1 si propagherebbe a tutti i token successivi e il training sarebbe instabile. Il teacher forcing garantisce che ogni step sia indipendente dagli errori precedenti.

**Exposure bias**

Il gap tra training e inferenza:

```
Training:   decoder vede sempre ground truth; condizioni perfette
Inferenza:  decoder vede le proprie predizioni; può propagare errori
```

Su sequenze corte l'effetto è trascurabile. Su sequenze lunghe può causare degenerazione dell'output.

---

### Inferenza — Generazione autoregressiva

A tempo di inferenza il decoder genera token per token usando le proprie predizioni:

```
Step 1: input [BOS]                 -> predice "ciao"
Step 2: input [BOS] ciao            -> predice "come"
Step 3: input [BOS] ciao come       -> predice "stai"
Step 4: input [BOS] ciao come stai  -> predice [EOS] -> stop
```

L'encoder viene eseguito **una sola volta** e il suo output rimane fisso per tutti i passi di generazione.

---

## 2. Differenze con Encoder-Only

|                    | Encoder-only          | Encoder-decoder           |
|--------------------|-----------------------|---------------------------|
| Goal               | capire il testo       | trasformare il testo      |
| Output             | dimensione fissa      | sequenza variabile        |
| Target nel training| un numero (etichetta) | un'intera sequenza        |
| Inferenza          | un forward pass       | un forward pass per token |
| Task tipici        | classificazione, NER  | traduzione, riassunto |

---

### Differenze nel preprocessing

**Encoder-only:**
```python
# un solo input, un'etichetta
input_ids      = [CLS] tokens [PAD]
attention_mask = [1, 1, 1, 0, 0]
label          = 1
```

**Encoder-decoder:**
```python
# due sequenze: sorgente e target
input_ids      = [token sorgente]
attention_mask = [maschera sorgente]
labels         = [token target]  # tipicamente -100 sulle posizioni di padding
```

### Il -100 sulle labels

`CrossEntropyLoss` ha `ignore_index=-100` per default. Ogni posizione delle labels con valore `-100` viene ignorata nel calcolo della loss — il modello non viene penalizzato per non aver predetto token di padding.

```python
labels = [178, 34, 56, -100, -100, -100]
#                       ↑ padding -> ignorato dalla loss
```

---

## 3. HuggingFace Transformers — classi usate

### AutoTokenizer

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")
```

Carica automaticamente il tokenizer corretto per il modello specificato. Per T5, gestisce sia l'input che il target con lo stesso vocabolario.

**Parametro `text_target`** — tokenizza input e target in una sola chiamata:

```python
model_inputs = tokenizer(
    inputs,
    text_target=targets,   # target tokenizzato automaticamente
    truncation=True,
    max_length=512,
    padding=False,
)
# model_inputs["labels"] contiene gli input_ids del target
```

**`batch_decode`** — converte una lista di sequenze di token in stringhe:

```python
testi = tokenizer.batch_decode(predictions, skip_special_tokens=True)
# skip_special_tokens=True rimuove [PAD], [EOS], [BOS]
```

---

### AutoModelForSeq2SeqLM

```python
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

Carica T5 con la testa di generazione seq2seq. A differenza di `AutoModelForSequenceClassification` (usato per DistilBERT), non aggiunge una nuova testa — T5 è già pre-addestrato per la generazione.

| Classe | Quando usarla |
|--------|---------------|
| `AutoModelForSequenceClassification` | classificazione (encoder-only) |
| `AutoModelForSeq2SeqLM` | generazione condizionata (encoder-decoder) |
| `AutoModelForCausalLM` | generazione libera (decoder-only) |

---

### DataCollatorForSeq2Seq

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)
```

Versione seq2seq del `DataCollatorWithPadding` usato per DistilBERT. La differenza:

- `DataCollatorWithPadding`: padda solo `input_ids` e `attention_mask`
- `DataCollatorForSeq2Seq`: padda anche le `labels`, usando `-100` per il padding

Richiede il modello (`model=model`) per conoscere il `pad_token_id` corretto.

---

### Seq2SeqTrainingArguments e Seq2SeqTrainer

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
```

Versioni seq2seq di `TrainingArguments` e `Trainer`. La differenza principale:

**`predict_with_generate=True`**

```python
Seq2SeqTrainingArguments(
    ...
    predict_with_generate=True,
)
```

Dice al Trainer di usare `model.generate()` durante la valutazione invece di prendere l'argmax dei logit.

```
Senza predict_with_generate:
  Trainer -> argmax(logits) -> token per posizione indipendentemente
  Ogni posizione è indipendente -> sequenza non coerente -> ROUGE sbagliato

Con predict_with_generate:
  Trainer -> model.generate() -> generazione autoregressiva
  Ogni token dipende dai precedenti -> sequenza coerente -> ROUGE corretto
```

---

## 4. Il dataset e la tokenizzazione

### CNN/DailyMail — struttura piatta

```python
dataset = load_dataset("cnn_dailymail", "3.0.0")
# colonne: article, highlights, id

esempio = dataset["train"][0]
# {
#   "article":    "LONDON, England...",
#   "highlights": "Harry Potter gets £20M...",
#   "id":         "42c027..."
# }
```

Input: `article` con prefisso `"summarize: "`  
Target: `highlights`

---

### Opus Books — struttura annidata

```python
dataset = load_dataset("opus_books", "en-it")
# colonne: id, translation

esempio = dataset["train"][0]
# {
#   "id": "0",
#   "translation": {"en": "hello", "it": "ciao"}
# }
```

La colonna `translation` contiene un dizionario — non è accessibile direttamente con `examples["it"]`. Serve:

```python
inputs  = [t["en"] for t in examples["translation"]]
targets = [t["it"] for t in examples["translation"]]
```

---

### remove_columns

```python
dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
```

Rimuove tutte le colonne originali dopo la tokenizzazione. Il Trainer si aspetta solo `input_ids`, `attention_mask`, `labels`; se lasci `article` o `highlights` il modello prova a usarle e va in crash.

---

## 5. Training — componenti chiave

### Flusso completo

```
config.py           iperparametri centralizzati
      ↓
data/dataset.py     carica, taglia, tokenizza
      ↓
model/model.py      carica T5 pre-addestrato
      ↓
training/metrics.py  definisce ROUGE
      ↓
training/trainer.py  configura Seq2SeqTrainer
      ↓
train.py            mette tutto insieme, avvia il training
      ↓
plot.py             visualizza loss e ROUGE per epoca
```

---

### Lettura del trainer_state.json

Il Trainer salva automaticamente `trainer_state.json` in ogni checkpoint. Contiene `log_history` — lista di dizionari con loss e metriche per ogni step e ogni valutazione:

```json
{
  "log_history": [
    {"loss": 2.34, "step": 100, "epoch": 0.5},
    {"loss": 1.98, "step": 200, "epoch": 1.0},
    {"eval_loss": 1.87, "eval_rouge1": 0.31, "epoch": 1.0},
    ...
  ]
}
```

Per leggere la storia globale basta leggere il `trainer_state.json` dell'**ultimo** checkpoint — contiene tutto il training dall'inizio.

---

### Learning rate per T5

```python
LEARNING_RATE = 5e-5
```

Leggermente più alto rispetto a DistilBERT (`2e-5`) perché T5-small richiede un learning rate più aggressivo per convergere in poche epoche. Valori tipici per il fine-tuning di modelli seq2seq:

```
T5-small:  3e-4 -> 5e-5
T5-base:   1e-4 -> 3e-5
BART:      3e-5 -> 5e-6
```
---

## 7. T5 — text-to-text

### Il prefisso

T5 è addestrato su ogni task NLP in formato testo->testo. Il prefisso dice al modello quale task eseguire:

```
"summarize: "                      -> summarization
"translate English to Italian: "   -> traduzione
"simplify: "                       -> semplificazione
"question: ... context: ..."        -> question answering
```

Senza prefisso i risultati peggiorano significativamente — il modello non sa quale comportamento adottare.

---

### Dimensioni disponibili

| Modello   | Parametri | Uso consigliato |
|-----------|-----------|-----------------|
| T5-small  | 60M       | CPU, demo, lezione |
| T5-base   | 220M      | GPU, esperimenti |
| T5-large  | 770M      | GPU dedicata |
| T5-3B     | 3B        | multi-GPU |
| T5-11B    | 11B       | infrastruttura cloud |

Per il corso usiamo **T5-small**: gira su CPU in tempi ragionevoli con dataset ridotti.

---

### MAX_INPUT_LENGTH vs MAX_TARGET_LENGTH

```python
MAX_INPUT_LENGTH  = 512   # lunghezza encoder
MAX_TARGET_LENGTH = 128   # lunghezza decoder
```

Il target è più corto dell'input perché:

- un riassunto è più corto dell'articolo originale
- una traduzione ha lunghezza simile all'originale ma raramente la supera
- limitare la lunghezza del target riduce il tempo di generazione

---

## 8. Struttura del progetto

```
t5_project/
├── config.py              ← iperparametri + configurazione task
├── data/
│   └── dataset.py         ← caricamento, tokenizzazione, select()
├── model/
│   └── model.py           ← carica AutoModelForSeq2SeqLM
├── training/
│   ├── metrics.py         ← compute_metrics con ROUGE
│   └── trainer.py         ← Seq2SeqTrainer + LogCallback
├── train.py               ← entry point
├── plot.py                ← visualizzazione loss e ROUGE
└── predict.py             ← inferenza su nuove frasi
```

### Come cambiare task

Modifica una sola riga in `config.py`:

```python
TASK = "summarization"   # -> cnn_dailymail, article/highlights
TASK = "translation"     # -> opus_books, en/it (struttura annidata)
TASK = "simplification"  # -> wiki_auto, normal/simple
```

Tutto il resto del codice si adatta automaticamente.

---

## 9. Domande frequenti

**Perché il Trainer usa Seq2Seq e non quello normale?**

`Seq2SeqTrainer` ha `predict_with_generate=True` — usa `model.generate()` durante la valutazione invece dell'argmax dei logit. Senza questo le metriche vengono calcolate su sequenze generate in modo non autoregressivo, il che non riflette le performance reali del modello.

---

**Perché -100 sulle labels e non 0?**

`0` potrebbe essere un indice valido nel vocabolario. `-100` è impossibile come indice (i vocabolari hanno sempre indici positivi) e corrisponde all'`ignore_index` default di `CrossEntropyLoss`.

---

**Perché select() va fatto prima di map()?**

`map()` itera su tutto il dataset. CNN/DailyMail ha 287k esempi — tokenizzarli tutti richiede ~10 minuti. Con `select(range(1000))` prima del `map()`, tokenizzi solo 1000 esempi in pochi secondi.

---

**Perché il ROUGE-2 è spesso basso?**

ROUGE-2 misura la sovrapposizione di coppie di parole consecutive. Due frasi possono avere molte parole in comune ma organizzate diversamente — "Harry Potter gets fortune" vs "gets fortune Harry Potter" ha ROUGE-1 alto e ROUGE-2 basso.

---

**Qual è la differenza tra trainer_state.json nei checkpoint?**

Ogni `trainer_state.json` contiene la storia **completa** del training dall'inizio, non solo dall'ultimo checkpoint. Leggere il `trainer_state.json` dell'ultimo checkpoint è sufficiente per vedere l'andamento globale.

---

**Perché opus_books ha struttura annidata?**

È una scelta del dataset su HuggingFace — i dataset di traduzione spesso raggruppano le coppie linguistiche in un dizionario `translation` per chiarezza. Non è una regola universale — altri dataset di traduzione possono avere struttura piatta. Controlla sempre con `dataset["train"].column_names` e `dataset["train"][0]` prima di scrivere il codice.
