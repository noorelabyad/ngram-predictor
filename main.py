from dotenv import load_dotenv
import argparse
import os
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel 
from src.inference.predictor import Predictor


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

def main():
    """Run the desired steps of the pipeline based on command line arguments."""

    # Uses the test data (runs on smaller raw data sample) instead of the main config and data
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env.test"))

    # Uses the full data
    # load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))

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
    else:
        print("Invalid step. Choose from: dataprep, model, inference, all.")

if __name__ == "__main__":
    main()
