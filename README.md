# LLM_Course26

Benvenuto nel repository del corso **LLM & AI Generativa**.

---

## Come iniziare

### 1. Crea un account GitHub
Se non ce l'hai già, registrati su [github.com](https://github.com).

### 2. Fai fork della repo
Vai sulla pagina del repo del corso e clicca **Fork** in alto a destra.
Scegli il tuo account come destinazione: GitHub creerà una copia del repo nel tuo profilo.

### 3. Clona il tuo fork
```bash
git clone https://github.com/TUO-USERNAME/LLM_Course26.git
cd LLM_Course26
```

### 4. Collega il repo del corso come upstream
```bash
git remote add upstream https://github.com/Giuliaqqrd/LLM_Course26.git
```

Verifica che sia andato a buon fine:
```bash
git remote -v
```

Dovresti vedere:
```
origin    https://github.com/TUO-USERNAME/LLM_Course26.git (fetch)
origin    https://github.com/TUO-USERNAME/LLM_Course26.git (push)
upstream  https://github.com/Giuliaqqrd/LLM_Course26.git (fetch)
upstream  https://github.com/Giuliaqqrd/LLM_Course26.git (push)
```

### 5. Crea e attiva l'ambiente virtuale

Scegli uno dei due metodi — non usarli entrambi.

**Opzione A — venv** (più semplice, nessuna installazione aggiuntiva)
```bash
# crea il venv
python -m venv .venv

# attiva su Mac/Linux
source .venv/bin/activate

# attiva su Windows
.venv\Scripts\activate
```

**Opzione B — Miniconda** 

Se non hai Miniconda, scaricalo da [docs.anaconda.com/miniconda](https://docs.anaconda.com/miniconda/).

```bash
# crea l'ambiente
conda create -n llmcourse python=3.12

# attiva
conda activate llmcourse
```

---

> In entrambi i casi verifica di aver attivato l'ambiente prima di installare
> le dipendenze e prima di ogni sessione di lavoro.
> Il nome dell'ambiente apparirà tra parentesi nel terminale:
> `(.venv)` oppure `(llmcourse)`

### 6. Installa le dipendenze
```bash
pip install -r requirements.txt
```
#### Aggiornare requirements
```
pip freeze > requirements.txt
```

# Lavoraro e aggiornamento della propria fork

### 1. Assicurati di aver salvato il tuo lavoro
```bash
git add .
git commit -m "salvataggio prima di aggiornare"
```

### 2. Scarica gli aggiornamenti dal repo del corso
```bash
git pull upstream main
```

### 3. Carica gli aggiornamenti nel tuo fork
```bash
git push origin main
```

---

> Fai sempre questo procedimento **prima di ogni lezione**,
> non durante! Eviti conflitti mentre stai lavorando sui file.

---

## Se git pull dà un conflitto

Succede se hai modificato un file che è stato aggiornato dal proprietario della repo.
La soluzione più semplice:

```bash
# metti da parte le tue modifiche
git stash

# scarica gli aggiornamenti
git pull upstream main

# riprendi le tue modifiche
git stash pop
```
## Workflow consigliato — branch separato

Invece di lavorare direttamente su `main`, ti consiglio di creare
un branch separato chiamato `dev`. In questo modo `main` rimane
sempre sincronizzato con la repo del corso senza conflitti.

```
main   <- aggiornato dal professore, non si tocca
dev <- dove completi gli esercizi e sviluppi il progetto
```

Seguendo i passaggi della sezione precedente:

```bash
# torna su main
git checkout main

# scarica gli aggiornamenti
git pull upstream main

# torna su dev
git checkout dev

# porta i nuovi file su dev
git merge main
```

### Durante e dopo ogni lezione

Lavora normalmente sui file. Quando vuoi salvare:

```bash
git add .
git commit -m "lezione 1: completato tokenizer e data"
git push origin dev
```

---

# WORKFLOW DI GRUPPO

Se lavorate in gruppo sul progetto finale, uno di voi fa da **proprietario**
della repo e gli altri diventano **collaboratori**.

---

### Setup — chi fa da proprietario

Vai sul tuo fork su GitHub e aggiungi il tuo compagno come collaboratore.

Il compagno riceve una email di invito — deve accettarla prima di procedere.

---

### Setup — chi fa da collaboratore

Non fare fork nuovamente. Clona direttamente la repo del proprietario:

```bash
git clone https://github.com/USERNAME-PROPRIETARIO/LLM_Course26.git
cd LLM_Course26
```
---

### Workflow quotidiano

Ognuno lavora su un branch con il proprio cognome:

```bash
# studente A
git checkout -b rossi

# studente B
git checkout -b bianchi
```

Quando vuoi salvare e condividere il tuo lavoro:

```bash
git add .
git commit -m "aggiunto training loop"
git push origin rossi        # studente A
git push origin bianchi      # studente B
```

---

### Integrare il lavoro del compagno

Quando vuoi portare sul tuo branch le modifiche del compagno:

```bash
# scarica gli aggiornamenti da GitHub
git fetch origin

# porta il lavoro del compagno sul tuo branch
git merge origin/bianchi     # studente A porta il lavoro di B
git merge origin/rossi       # studente B porta il lavoro di A
```

---

### Prima di ogni lezione

Il **proprietario** scarica i nuovi file del professore e li condivide:

```bash
git checkout main
git pull upstream main
git push origin main
git checkout TUO-BRANCH
git merge main
```

Il **collaboratore** scarica gli aggiornamenti dal repo del proprietario:

```bash
git checkout main
git pull origin main
git checkout TUO-BRANCH
git merge main
```

---

### Regola d'oro — evitare i conflitti

I conflitti nascono quando due persone modificano le stesse righe
dello stesso file contemporaneamente. Per evitarli: 
dividetevi i file prima di iniziare a lavorare; è la cosa più
importante per una collaborazione senza problemi.

### Consegna finale - progetto individuale

Quando il progetto è completato, porta tutto su `main` e caricalo su GitHub:

```bash
# porta il tuo lavoro su main
git checkout main
git merge TUO-BRANCH

# carica su GitHub
git push origin main
```
---

### Consegna finale — progetto di gruppo

Il **proprietario** porta il lavoro di entrambi su `main`:

```bash
# porta il lavoro di entrambi
git checkout main
git merge rossi
git merge bianchi

# carica su GitHub
git push origin main
```

Il **collaboratore** non deve fare nulla: deve essere condivisa soltanto la repo del proprietario (invitare Giuliaqqrd come collaboratore).

