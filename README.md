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
- Check requirements.txt to install dependencies

## Setup
1. Clone the repo
2. Create and Activate Anaconda Environment
3. Install Dependencies 
4. Populate config/.env
   - Required Variables : 
    - TRAIN_TOKENS (path *string*)
    - EVAL_TOKENS (path *string*)
    - MODEL (path *string*)
    - VOCAB (path *string*)
    - UNK_THRESHOLD (*integer*)
    - TOP_K (*integer*)
    - NGRAM_ORDER (*integer*)
5. Download raw .txt files into the correct folders 

## Usage

## Project Structure 

ngram-predictor/
- config/
  - .env
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