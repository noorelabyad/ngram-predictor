from dotenv import load_dotenv
import argparse
import os
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel 
from src.inference.predictor import Predictor
load_dotenv(dotenv_path="config/.env")

def main():

    parser = argparse.ArgumentParser(description="run upto which step")
    parser.add_argument("--step", default="inference", help="run upto which step")
    args = parser.parse_args()

    if args.step == "dataprep":

        normalizer = Normalizer(
            folder_path=os.getenv("TRAIN_RAW_DIR"), 
            output_file=os.getenv("TRAIN_TOKENS")
        )

        normalizer.main()


    if args.step == "model":

        ngram_model = NGramModel (
            token_file=os.getenv("TRAIN_TOKENS"),
            model_path=os.getenv("MODEL_PATH"),
            vocab_path=os.getenv("VOCAB_PATH"),
            unk_threshold=int(os.getenv("UNK_THRESHOLD")),
            ngram_order=int(os.getenv("NGRAM_ORDER"))
        )

        ngram_model.main()

    if args.step == "inference":

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

    
    

if __name__ == "__main__":
    main()
