import os

class ngram_model:

    def __init__(self,  token_file, model_path, vocab_path, unk_threshold, ngram_order) :    
        self.token_file = token_file
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.unk_threshold = unk_threshold
        self.ngram_order = ngram_order

    def build_vocab(self) :
        """Build vocabulary; apply UNK threshold"""
        pass
        
    def build_counts_and_probabilities(self, token_file) :
        """Count all n-grams at orders 1 through NGRAM_ORDER and compute MLE probabilities. 
        Probabilities depend on counts, so they are computed together to avoid hidden ordering bugs."""
        pass

    @staticmethod
    def lookup(context) :
        """Backoff lookup: try the highest-order context first, fall back to lower orders down to 1-gram. 
        Return a dict of {word: probability} from the highest order that matches. Return empty dict if no match at any order. 
        This is the single source of backoff logic in the project."""
        pass

    def save_model(self) :
        """Save all probability tables to model.json"""
        pass

    def save_vocab(self) :
        """Save vocabulary to vocab.json"""
        pass

    def load(self) :
        """Load model.json and vocab.json into the instance. Called once in main() before passing the model to Predictor"""
        pass
    
    def main(self) :
        pass

def main() :
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../../config/.env")
    ngram_model1 = ngram_model(
        token_file=os.getenv("TRAIN_TOKENS"),
        model_path=os.getenv("MODEL_PATH"),
        vocab_path=os.getenv("VOCAB_PATH"),
        unk_threshold=int(os.getenv("UNK_THRESHOLD")),
        ngram_order=int(os.getenv("NGRAM_ORDER"))
    )
    ngram_model1.main()



if __name__ == "__main__":
    main()
        