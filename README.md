# Next-Word Prediction System using N-GRAM Language Model 

## Description

This project is a complete implementation of a next‑word prediction system built using a classic n‑gram language model. It processes a collection of Sherlock Holmes novels to learn how words naturally follow one another, then uses that statistical knowledge to predict the most likely next words a user might type.
The system walks through the real workflow behind a lightweight NLP model:
- Preparing and cleaning text data
- Building and training an n‑gram model
- Generating predictions from user input with backoff logic
- Providing a command‑line interface for interactive use

## Requirements
- Python Version : 3.14.3
- Check `requirements.txt` to install dependencies

## Setup
1. Clone the repo
2. Create and Activate Anaconda Environment
3. Install Dependencies 
4. Populate config/.env 
   - Required Variables : 
    - TRAIN_TOKENS : Path to the folder that contains the .txt files to be used for training the model
    - MODEL_PATH : Path to .json file where the model (dictionary) containing the ngram probablities given contexts will be stored
    - VOCAB_PATH : Path to .json file where the vocabulary (dictionary) will be stored after extraction from training data
    - UNK_THRESHOLD : The minimum number of times a word must appear in the training data to be included in the vocab of the model
    - TOP_K : The number of top suggestions for the next most probable words suggested by the model to the user
    - NGRAM_ORDER : The maximum ngram order that the model will calculate and store the probabilities for
5. Download raw .txt files into the correct folders (default setup: data/raw/train/)

## Usage

This project provides a command-line interface to run different stages of the language modeling pipeline:
- Data preparation
- Model training
- Inference (text prediction)
- Or the entire pipeline end‑to‑end

All configuration (paths, model parameters, and hyperparameters) is loaded from environment variables defined in `config\.env`.

 #### There are two options for running the code :
  1. Loading the `config\.env.test` and running on the smaller test data (faster for testing and debugging) or 
  2. Loading the `config\.env` and running on the full data (takes much more time).
- You can switch between the two by commenting/uncommenting the corresponding `load_dotenv()` lines in the
main() function in `main.py`.
---

### Running the Program (CLI)

All functionality is accessed through `main.py` using the `--step` argument.

```bash
python main.py --step <STEP_NAME>
```

If --step is not provided, the program defaults to running inference.

#### **Available Steps**

**dataprep** : Normalizes raw training text files and writes tokenized output to disk.

```bash
python main.py --step dataprep
```

This step:
- Reads raw text from `TRAIN_RAW_DIR`
- Applies normalization
- Saves normalized tokens to `TRAIN_TOKENS`

---

**model** : Builds the vocabulary, computes n‑gram counts and probabilities, and saves them to disk.

```bash
python main.py --step model
```
This step:
- Reads tokens from `TRAIN_TOKENS`
- Builds an n‑gram model with order `NGRAM_ORDER`
- Applies unknown token threshold `UNK_THRESHOLD`
- Saves the model to `MODEL_PATH`
- Saves the vocabulary to `VOCAB_PATH`

---

**inference** : Loads the trained model and starts an interactive prediction session.

```bash
python main.py --step inference
```
This step:
- Loads the trained n‑gram model from disk
- Instantiates the predictor
- Prompts the user to enter a sentence
- Outputs the top‑`k` predicted next tokens (controlled by `TOP_K`)
- Continues until the user types `quit`

---

**all** : Runs the full pipeline sequentially (data preparation, model training, and inference).

```bash
python main.py --step all
```
This step:
- Normalizes raw training data
- Trains the n‑gram model
- Starts the interactive inference loop
---
### Running the Predictor UI

The Streamlit UI provides an interactive browser interface for next‑word prediction.
Streamlit must be launched using streamlit run, not python.
```bash 
streamlit run src/ui/app.py 
```
This will:
- Load the trained n‑gram model
- Open a local browser window
- Allow users to type text and view top‑k next‑word predictions

---


## Project Structure 

ngram-predictor/
- config/
  - .env (points to 'data/' for complete training data)
  - .env.test (points to 'data_test/' for a small subset of the training data for testing)
- data_test/ 
  - raw/
    - train/
  - processed/
    - train_tokens.txt
  - model/
    - model.json
    - vocab.json
- data/
  - raw/
    - train/
  - processed/
    - train_tokens.txt
  - model/
    - model.json
    - vocab.json
- src/
  - data_prep/
    - normalizer.py
  - model/
    - ngram_model.py
  - inference/
    - predictor.py
- main.py
- .gitignore
- requirements.txt
- README.md