# BART e T5 — Reference Guide

> Guida di riferimento per il corso LLM & AI Generativa  
> Lezione 3: Architetture Encoder-Decoder a confronto

---

## Indice

1. [Panoramica — cosa hanno in comune](#1-panoramica--cosa-hanno-in-comune)
2. [T5 — Text-to-Text Transfer Transformer](#2-t5--text-to-text-transfer-transformer)
3. [BART — Denoising Autoencoder](#3-bart--denoising-autoencoder)
4. [Confronto diretto](#4-confronto-diretto)
5. [Pre-training a confronto](#5-pre-training-a-confronto)
6. [Fine-tuning — differenze pratiche](#6-fine-tuning--differenze-pratiche)
7. [Quando usare quale](#7-quando-usare-quale)
8. [Dimensioni e varianti disponibili](#8-dimensioni-e-varianti-disponibili)
9. [Domande frequenti](#9-domande-frequenti)

---

## 1. Panoramica — cosa hanno in comune

T5 e BART sono entrambi modelli **encoder-decoder** basati sull'architettura Transformer. Condividono la stessa struttura fondamentale:

```
Input sequence
      ↓
Encoder (bidirectional self-attention × N)
      ↓
Encoded representation
      ↓
Decoder (masked self-attention + cross-attention × N)
      ↓
Output sequence (generata token per token)
```

Entrambi sono stati introdotti nel 2019-2020 e sono ancora oggi tra i modelli di riferimento per i task di generazione condizionata. La differenza fondamentale non è nell'architettura ma nel **modo in cui sono stati pre-addestrati**.

---

## 2. T5 — Text-to-Text Transfer Transformer

### Il paper

Raffel et al., 2019 — *"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"* (Google)

### L'idea centrale

T5 unifica **tutti** i task NLP in un unico formato testo→testo. Ogni task viene formulato come: dai in input una stringa, ottieni in output una stringa.

```
Traduzione:       "translate English to German: Hello"    ->"Hallo"
Summarization:    "summarize: Scientists discovered..."   ->"Researchers find..."
Classificazione:  "sst2 sentence: The movie was great"   ->"positive"
QA:               "question: Who is Harry? context: ..."  ->"Daniel Radcliffe"
```

Questo approccio ha un vantaggio enorme: **un solo modello per task infiniti**. Non serve cambiare architettura, aggiungere teste di classificazione o modificare il codice — basta cambiare il prefisso.

### Il prefisso

Il prefisso è un'istruzione testuale che dice al modello quale task eseguire:

```python
"summarize: "                      summarization
"translate English to Italian: "   traduzione
"simplify: "                       semplificazione
"question: ... context: ..."       question answering
```

Durante il pre-training T5 ha visto migliaia di task diversi, ognuno con il suo prefisso. Il modello ha imparato che il prefisso determina il comportamento. Senza prefisso i risultati peggiorano significativamente.

### Il pre-training

T5 è pre-addestrato con un obiettivo chiamato **span corruption** (corruzione di span):

```
Testo originale:
"The quick brown fox jumps over the lazy dog"

Testo corrotto (input encoder):
"The quick <X> jumps over <Y> dog"

Target (input decoder):
"<X> brown fox <Y> the lazy"
```

Alcuni span (sequenze consecutive di parole) vengono sostituiti con token speciali come `<X>`, `<Y>`. Il modello deve ricostruire i span mancanti.

Questo è simile al Masked Language Modeling di BERT ma più generale — invece di mascherare singoli token, si mascherano sequenze intere, e il modello deve generarle (non solo classificarle).

### Il dataset di pre-training

T5 è pre-addestrato su **C4** (Colossal Clean Crawled Corpus) — 750GB di testo pulito estratto da Common Crawl (web crawling). È uno dei dataset più grandi mai usati per il pre-training di un modello linguistico all'epoca.

### Architettura specifica

T5 usa una versione leggermente modificata del Transformer originale:

```
Layer normalization:  Pre-LN (prima dell'attention) invece di Post-LN
Positional encoding:  Relative position bias invece di sinusoidale
Activation:           ReLU nel feedforward
Feed-forward:         Gated variant in T5 v1.1
```

Il **relative position bias** è una delle innovazioni chiave di T5: invece di codificare la posizione assoluta di ogni token, codifica la distanza relativa tra coppie di token. Questo permette al modello di generalizzare meglio a sequenze di lunghezza diversa da quelle viste durante il training.

---

## 3. BART — Denoising Autoencoder

### Il paper

Lewis et al., 2019 — *"BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"* (Facebook AI)

### L'idea centrale

BART è un **denoising autoencoder**: viene addestrato a ricostruire testo corrotto. L'encoder riceve testo con vari tipi di "rumore", il decoder deve ricostruire il testo originale.

```
Testo originale:
"The scientists discovered a new species of fish"

Testo corrotto (input encoder):
"scientists The discovered fish species new a of"  <- parole mischiate

Target (input decoder):
"The scientists discovered a new species of fish"  <- testo originale
```

### I tipi di corruzione

BART è stato addestrato con cinque tipi di trasformazioni:

**1. Token masking**: come BERT, singoli token vengono sostituiti con `[MASK]`:
```
"The [MASK] discovered a [MASK] species"
```

**2. Token deletion**: token vengono eliminati completamente (il modello deve capire dove mancano):
```
"The scientists a new of fish"
```

**3. Text infilling** — uno span di token viene sostituito con un singolo `[MASK]` (il modello deve generare quanti token servono):
```
"The [MASK] fish"   <- un solo mask per "scientists discovered a new species of"
```

**4. Sentence permutation**: le frasi del documento vengono mischiate:
```
"A new species of fish. The scientists discovered it. In the Pacific Ocean."
->"In the Pacific Ocean. The scientists discovered it. A new species of fish."
```

**5. Document rotation**: il documento viene ruotato intorno a un token casuale:
```
"discovered a new species of fish. The scientists"
```

La combinazione di questi tipi di corruzione, specialmente text infilling e sentence permutation, rende BART particolarmente bravo nella generazione di testo lungo e coerente.

### Perché BART è bravo per summarization

La sentence permutation è particolarmente utile per la summarization: il modello impara a capire il documento indipendentemente dall'ordine delle frasi, e a riorganizzare le informazioni in modo logico. Questo è esattamente quello che serve per produrre un buon riassunto.

### Il dataset di pre-training

BART è pre-addestrato su una combinazione di:
- **Books**: libri digitalizzati
- **CC-News**: articoli di giornale da Common Crawl
- **OpenWebText**: testo web (simile al dataset di GPT-2)
- **Stories**: racconti e storie

In totale circa 160GB di testo, molto meno di T5, ma più focalizzato su testo di qualità.

### Architettura specifica

BART usa l'architettura Transformer originale con poche modifiche:

```
Layer normalization:  Post-LN (dopo l'attention) — come il paper originale
Positional encoding:  Learned absolute (come GPT e RoBERTa)
Activation:           GeLU nel feedforward
Dimensioni:           d_model=1024, nhead=16, num_layers=12 (BART-large)
```

Una caratteristica importante: l'encoder di BART è identico a **RoBERTa**: il miglior encoder-only dell'epoca. Questo significa che BART eredita le ottime capacità di comprensione del testo di RoBERTa, combinate con le capacità generative del decoder.

### Nessun prefisso

A differenza di T5, BART non usa prefissi testuali. Il task viene determinato dal fine-tuning, durante il fine-tuning per summarization il modello impara che il suo compito è riassumere, senza bisogno di un'istruzione esplicita:

```python
# T5
input = "summarize: The scientists discovered..."

# BART
input = "The scientists discovered..."   # nessun prefisso
```

---

## 4. Confronto diretto

| Caratteristica | T5 | BART |
|----------------|-----|------|
| Pre-training | Span corruption | Denoising (5 tipi) |
| Prefisso task | Sì, obbligatorio | No |
| Dataset pre-training | C4 (750GB) | Books + News + Web (160GB) |
| Encoder | Transformer modificato | Identico a RoBERTa |
| Positional encoding | Relative bias | Learned absolute |
| Layer norm | Pre-LN | Post-LN |
| Dimensioni base | 220M (T5-base) | 139M (BART-base) |
| Punto di forza | Versatilità (molti task) | Qualità generazione (summarization) |
| Punto debole | Richiede prefisso | Meno versatile su task diversi |

---

## 5. Pre-training a confronto

### T5 — Span Corruption

```
Prima:  "The quick brown fox jumps over the lazy dog"
         ↓
Corruzione (15% dei token in span):
Input:  "The quick <X> jumps <Y> lazy dog"
Target: "<X> brown fox <Y> over the"
```

Il modello impara a:
- capire il contesto intorno ai gap
- generare span di lunghezza variabile
- non ha mai visto il testo originale durante il pre-training

### BART — Denoising

```
Prima:  "The scientists discovered a new species"
         ↓
Corruzione (text infilling + sentence permutation):
Input:  "discovered [MASK] The species"
Target: "The scientists discovered a new species"
```

Il modello impara a:
- ricostruire il testo originale da versioni corrotte
- capire la struttura del documento indipendentemente dall'ordine
- generare testo fluente e coerente

### Impatto sulle performance

La differenza nel pre-training si riflette nelle performance sui task:

```
Summarization:
  BART-large:  ROUGE-1=44.16, ROUGE-2=21.28, ROUGE-L=40.90
  T5-large:    ROUGE-1=42.50, ROUGE-2=20.68, ROUGE-L=39.75
  ->BART migliore per summarization

Traduzione EN→DE:
  T5-base:   BLEU=26.98
  BART-base: BLEU=21.34
  ->T5 migliore per traduzione

Classificazione (GLUE):
  T5-base:   accuracy media ~84%
  BART-base: accuracy media ~88%
  ->BART migliore per comprensione
```

---

## 6. Fine-tuning — differenze pratiche

### Codice — cosa cambia

La differenza nel codice è minima grazie ad `AutoModelForSeq2SeqLM`:

```python
# T5
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# BART — stessa classe, stesso codice
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
```

HuggingFace gestisce le differenze architetturali internamente; tu non devi sapere che BART usa Post-LN e T5 usa Pre-LN.

### Differenze negli iperparametri

```python
# T5-small
LEARNING_RATE     = 5e-5
TRAIN_BATCH_SIZE  = 8
MAX_INPUT_LENGTH  = 512

# BART-base
LEARNING_RATE     = 3e-5    # più conservativo — modello più grande
TRAIN_BATCH_SIZE  = 4       # batch più piccolo — più memoria
MAX_INPUT_LENGTH  = 1024    # BART supporta sequenze più lunghe
```

### Il prefisso in fase di inferenza

```python
# T5: prefisso obbligatorio anche a tempo di inferenza
input_text = "summarize: " + articolo
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs)

# BART: nessun prefisso
inputs = tokenizer(articolo, return_tensors="pt")
output = model.generate(**inputs)
```

### Generazione — parametri comuni

Entrambi usano `model.generate()` per la generazione a tempo di inferenza:

```python
output = model.generate(
    input_ids,
    max_new_tokens=128,       # lunghezza massima output
    num_beams=4,              # beam search con 4 beam
    early_stopping=True,      # fermati quando tutti i beam producono [EOS]
    no_repeat_ngram_size=3,   # evita ripetizioni di 3-grammi
    length_penalty=2.0,       # penalizza output troppo corti (>1) o lunghi (<1)
)
```

**Beam search**: invece di prendere sempre il token più probabile (greedy), mantiene le N sequenze più probabili in parallelo e sceglie quella con il punteggio complessivo più alto. Produce output di qualità superiore al greedy a costo di essere più lento.

---

## 7. Quando usare quale

### Usa T5 quando:

- Hai **task multipli**: un solo modello per summarization, traduzione, QA e classificazione
- Vuoi **addestrare su task nuovi** con istruzioni testuali
- Hai **risorse limitate**: T5-small (60M) è molto più leggero di BART-base (139M)
- Stai costruendo un sistema **instruction-following** dove il prefisso guida il comportamento

### Usa BART quando:

- Il tuo task principale è **summarization**: BART è stato specificamente ottimizzato per questo
- Hai bisogno di **generazione di testo lunga** e coerente
- Vuoi sfruttare le capacità di **comprensione di RoBERTa** nell'encoder
- Non vuoi gestire i prefissi

---

## 8. Dimensioni e varianti disponibili

### T5

| Modello | Parametri | Layers | d_model | Note |
|---------|-----------|--------|---------|------|
| t5-small | 60M | 6 | 512 | ottimo per CPU e demo |
| t5-base | 220M | 12 | 768 | buon compromesso |
| t5-large | 770M | 24 | 1024 | GPU necessaria |
| t5-3b | 3B | 24 | 1024 | multi-GPU |
| t5-11b | 11B | 24 | 1024 | infrastruttura cloud |
| t5-v1_1-* | varie | — | — | versione migliorata, stesse dimensioni |
| flan-t5-* | varie | — | — | T5 con instruction tuning su 1800+ task |

### BART

| Modello | Parametri | Layers enc | Layers dec | Note |
|---------|-----------|------------|------------|------|
| facebook/bart-base | 139M | 6 | 6 | buon punto di partenza |
| facebook/bart-large | 406M | 12 | 12 | GPU necessaria |
| facebook/bart-large-cnn | 406M | 12 | 12 | fine-tuned su CNN/DailyMail |
| facebook/bart-large-xsum | 406M | 12 | 12 | fine-tuned su XSum |
| facebook/mbart-large-50 | 610M | 12 | 12 | multilingue, 50 lingue |

`facebook/bart-large-cnn` è particolarmente interessante perchè è già fine-tuned per summarization, puoi usarlo direttamente senza training e confrontare le sue performance con un modello fine-tuned.

---

