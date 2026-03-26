import os
import sys
sys.path.append(os.getcwd())

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel

class Predictor:
    """module's responsibility: taking user input, normalizing it using the same steps as in training, mapping any out-of-vocab word to <UNK>, 
    and using the n-gram model to predict the top-k most likely next words."""
    
    def __init__(self, ngram_model, normalizer, top_k) :
        self.ngram_model = ngram_model
        self.normalizer = normalizer
        self.top_k = top_k

    def normalize(self, text) :
        """Normalize the input text using the same normalization steps as in training
        Parameter 'text' is the raw user input string. 
        Returns the normalized context string that will be used for prediction"""
        text = Normalizer.normalize(text).split()
        return " ".join(text[-(self.ngram_model.ngram_order - 1) :])

    def map_oov(self,context) :
        """Map any out-of-vocab word in the input string parameter 'context' to <UNK> and returns the mapped string."""
        return " ".join([word if word in self.ngram_model.vocab else "<UNK>" for word in context.split()])
    
    def predict_next(self, text) :
        """Given an input string parameter 'text', returns a list of the top-k most likely next words based on the n-gram model."""
        context = self.normalize(text)
        context = self.map_oov(context)
        next_words = self.ngram_model.lookup(context)
        likeliest_next_words = dict(sorted(next_words.items(), key=lambda item: item[1]))
        return list(likeliest_next_words.keys())[-self.top_k :][::-1]
    
    
def main() :

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env.test"))

    normalizer=Normalizer(
        folder_path=os.getenv("TRAIN_RAW_DIR"), 
        output_file=os.getenv("TRAIN_TOKENS")
        )
    normalizer.main()

    ngram_model=NGramModel(
        token_file=os.getenv("TRAIN_TOKENS"),
        model_path=os.getenv("MODEL_PATH"),
        vocab_path=os.getenv("VOCAB_PATH"),
        unk_threshold=int(os.getenv("UNK_THRESHOLD")),
        ngram_order=int(os.getenv("NGRAM_ORDER"))
        )
    ngram_model.main()

    predictor1 = Predictor(
        ngram_model=ngram_model,
        normalizer=normalizer,
        top_k=int(os.getenv("TOP_K"))
        )

    # Test prediction
    print(predictor1.predict_next("the game is"))
    print(predictor1.predict_next("zzz qqq"))

if __name__ == "__main__":
    main()

        