from dotenv import load_dotenv
import argparse
import os
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel 
from src.inference.predictor import Predictor
from src.evaluation.evaluator import Evaluator

""" There are two modes to run the code:
loading the config.env.test and running on the smaller test data (faster for testing and debugging)
or loading the config.env and running on the full data (takes more time but gives better results).
You can switch between the two by commenting/uncommenting the corresponding load_dotenv lines in the
main() function at the bottom of this file.
"""
"""This module is the main entry point to run the whole pipeline. 
It contains functions to run each step of the pipeline separately (data preparation, model training, and inference), 
as well as a function to run the whole pipeline end-to-end. 
The main() function uses command line arguments to determine which steps to run.

THE DEFAULT STEP IS INFERENCE; SO THAT THE ALREADY TRAINED MODEL CAN BE LOADED AND TESTED WITHOUT HAVING 
TO RE-RUN THE DATA PREP AND MODEL TRAINING STEPS WHICH TAKES A LONG TIME ON THE FULL DATA.
"""

def dataprep() :
    """Run the data preparation step: normalize raw text files and save the normalized tokens to a new file."""
    normalizer = Normalizer(
        folder_path=os.getenv("TRAIN_RAW_DIR"), 
        output_file=os.getenv("TRAIN_TOKENS")
    )

    normalizer.main()

def model() :
    """Run the model training step: build the vocab, the n-gram counts and probabilities, save them to json files, 
    and load them back into the instance."""
    ngram_model = NGramModel (
        token_file=os.getenv("TRAIN_TOKENS"),
        model_path=os.getenv("MODEL_PATH"),
        vocab_path=os.getenv("VOCAB_PATH"),
        unk_threshold=int(os.getenv("UNK_THRESHOLD")),
        ngram_order=int(os.getenv("NGRAM_ORDER"))
    )

    ngram_model.main()

def inference() :
    """ Instantiate Normalizer and NGramModel, load the model, and instantiate Predictor.
    Enter a loop that prompt the user for input, calls Predictor.predict_next(text, k), and prints the top-k predictions."""
    normalizer=Normalizer(
        folder_path=os.getenv("TRAIN_RAW_DIR"), 
        output_file=os.getenv("TRAIN_TOKENS")
        )
    ngram_model=NGramModel(
        token_file=os.getenv("TRAIN_TOKENS"),
        model_path=os.getenv("MODEL_PATH"),
        vocab_path=os.getenv("VOCAB_PATH"),
        unk_threshold=int(os.getenv("UNK_THRESHOLD")),
        ngram_order=int(os.getenv("NGRAM_ORDER"))
        )
    ngram_model.load()

    predictor = Predictor(
        ngram_model=ngram_model,
        normalizer=normalizer,
        top_k=int(os.getenv("TOP_K"))
        )
    
    while True:
        user_input = input("Type sentence (or type 'quit' to exit): ")

        if user_input.lower() == "quit":
            break

        print(predictor.predict_next(user_input))

def all() :
    """Runs the whole pipeline: data preparation, model training, and inference."""
    dataprep()
    model()
    inference()

def evaluate():
    """Run perplexity evaluation on the held-out corpus."""
    normalizer = Normalizer(
        folder_path=os.getenv("EVAL_RAW_DIR"),
        output_file=os.getenv("EVAL_TOKENS")
    )
    normalizer.main()

    ngram_model = NGramModel(
        token_file=os.getenv("TRAIN_TOKENS"),
        model_path=os.getenv("MODEL_PATH"),
        vocab_path=os.getenv("VOCAB_PATH"),
        unk_threshold=int(os.getenv("UNK_THRESHOLD")),
        ngram_order=int(os.getenv("NGRAM_ORDER"))
    )
    ngram_model.load()

    evaluator = Evaluator(
        model=ngram_model,
        normalizer=normalizer,
        eval_file=os.getenv("EVAL_TOKENS")
    )

 
    evaluator.run()

def main():
    """Run the desired steps of the pipeline based on command line arguments."""
    
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))

    parser = argparse.ArgumentParser(description="run upto which step")
    parser.add_argument("--step", default="inference", help="run upto which step")
    args = parser.parse_args()

    if args.step == "dataprep":
        dataprep()
    elif args.step == "model":
        model()
    elif args.step == "inference":
        inference()
    elif args.step == "all":
        all()
    elif args.step == "evaluate":
        evaluate()
    else:
        print("Invalid step. Choose from: dataprep, model, inference, all.")

if __name__ == "__main__":
    main()
