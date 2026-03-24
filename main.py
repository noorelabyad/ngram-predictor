from dotenv import load_dotenv
import argparse
import os
from src.data_prep.normalizer import normalizer
from src.model.ngram_model import NGramModel 
load_dotenv(dotenv_path="config/.env")

def main():

    parser = argparse.ArgumentParser(description="run upto which step")
    parser.add_argument("--step", default="dataprep", help="run upto which step")
    args = parser.parse_args()

    if args.step == "dataprep":
        normalizer1 = normalizer(folder_path=os.getenv("TRAIN_RAW_DIR"), output_file=os.getenv("TRAIN_TOKENS"))

        # Training
        normalizer1.main()

        # Normalizer Test
        print(normalizer.normalize("  Hello, World! This is a test. 123   "))

    if args.step == "model":
        ngram_model1 = NGramModel (
            token_file=os.getenv("TRAIN_TOKENS"),
            model_path=os.getenv("MODEL_PATH"),
            vocab_path=os.getenv("VOCAB_PATH"),
            unk_threshold=int(os.getenv("UNK_THRESHOLD")),
            ngram_order=int(os.getenv("NGRAM_ORDER"))
        )

        # Trainig 
        ngram_model1.main()

        # Test lookup
        print(ngram_model1.lookup("hi across the"))
        print(ngram_model1.lookup("the"))
    

if __name__ == "__main__":
    main()
