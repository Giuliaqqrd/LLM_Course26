# TODO

# Obiettivo
Fine-tuning di T5 con un dataset per summarization in italiano: https://huggingface.co/datasets/ARTeLab/ilpost

### STEP 1
Usare data/check_dataset.py per capire come è strutturato il dataset e se possiede già gli split.

### STEP 2
Aggiornare il file di config:
```python
"summarization": {
        "name":       "ARTeLab/ilpost",
        "config":     None,           # nessuna sottoconfig
        "input_col":  "source",
        "target_col": "target",
        "prefix":     "summarize: ",
        "lang":       "it",
    }
```
Siccome config = None, bisogna gestire questo caso:
```python
# in load_and_tokenize()
if task_cfg["config"] is not None:
    dataset = load_dataset(task_cfg["name"], task_cfg["config"])
else:
    dataset = load_dataset(task_cfg["name"])
```
### STEP 3
Eseguire singolarmente tutti moduli per vedere che non ci siano errori e poi lanciare train.py. 
Per usare il dataset completo, modifica `config.py`:

```python
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES  = None
```
Potete lanciare l'addestramento in background in modo da poter continuare a lavorare su altre cose:

```
nohup python train.py > training.log 2>&1 &
```

### STEP 4
Usare plot.py per visualizzare l'andamento del training e poi provare a utilizzare il modello (predict.py) su dati mai visti. Si può scegliere un qualunque sito italiano di notizie e incollare parte di un articolo.
Di default predict usa l'ultimo checkpoint, ma si può passare manualmente quello desiderato:
```
python predict.py --model_dir checkpoints/summarization_t5-small_ilpost_lr5e-05_ep3_20260430
```