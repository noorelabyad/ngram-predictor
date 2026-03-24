import os
import json

class NGramModel :

    """module's responsibility: building, storing, 
    and exposing n-gram probability tables and backoff lookup across all orders from 1 up to NGRAM_ORDER"""

    def __init__(self,  token_file, model_path, vocab_path, unk_threshold, ngram_order, model=None, vocab=None) :    
        self.token_file = token_file
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.unk_threshold = unk_threshold
        self.ngram_order = ngram_order

    def build_vocab(self) :
        """Build vocabulary; collect all unique words; replace any word appearing fewer than UNK_THRESHOLD times"""
        with open(self.token_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = text.split()
        vocab = []
        for token in tokens :
                if tokens.count(token) < self.unk_threshold :
                    vocab.append("<UNK>")
                elif token not in vocab :
                    vocab.append(token)

        return vocab

    def build_counts_and_probabilities(self) :
        """Count all n-grams at orders 1 through NGRAM_ORDER and compute MLE probabilities. 
        Probabilities depend on counts, so they are computed together to avoid hidden ordering bugs."""

        
        # a window slides through scentences collecting ngrams and ngrams_plus_next_word
        # for e.g. for 3-grams, the window size is 4, and the first 3 words are the context and the 4th word is the next word.
        # probability of a word given a context = count(ngram) / count(ngram_minus_next_word) for n > 1, and count(ngram) / total_tokens for unigrams.
        # the probabilities are stored in a nested dict of dicts: {ngram_order: {ngram_minus_next_word: {next_word: probability}}} 
        # for all ngrams at all orders.

        with open(self.token_file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = text.split("\n")

        prob_dict = {}
        for n in range(1, self.ngram_order + 1) :
            prob_dict[f"{n}gram"] = {}
            for sentence in sentences  :
                for start in range(len(sentence.split()) - n + 1) :
                    ngram = []
                    for k in range(start, start + n ) :
                        if k < len(sentence.split()) :
                            ngram.append(sentence.split() [k])
                    ngram_minus_next_word = " ".join(ngram[:-1])
                    ngram_word = " ".join(ngram)
                    
                    probability= text.count(ngram_word)/text.count(ngram_minus_next_word) if n > 1 else text.count(ngram_word)/len(text.split())
                    if len(ngram_word.split()) == n : # for e.g. not to add 2-grams as 3-grams because that is how the sentence ended
                        if n > 1 :
                            if ngram_minus_next_word not in prob_dict[f"{n}gram"] :
                                prob_dict[f"{n}gram"][ngram_minus_next_word] = {}
                                prob_dict[f"{n}gram"][ngram_minus_next_word][ngram_word.split()[-1]] = probability
                            else :
                                prob_dict[f"{n}gram"][ngram_minus_next_word][ngram_word.split()[-1]] = probability
                        else :
                            if ngram_word not in prob_dict[f"{n}gram"] :
                                prob_dict[f"{n}gram"][ngram_word] = probability


        return prob_dict           

    def lookup(self, context) :
        """Backoff lookup: try the highest-order context first, fall back to lower orders down to 1-gram. 
        Return a dict of {word: probability} from the highest order that matches. Return empty dict if no match at any order. 
        This is the single source of backoff logic in the project."""
        while len(context.split()) > 0 :
            for n in range(self.ngram_order, 0, -1) :
                if context in self.model[f"{n}gram"] :
                    return self.model[f"{n}gram"][context]
            context = " ".join(context.split()[1:]) 
        return {}
    
    def save_model(self) :
        """Save all probability tables to model.json"""
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(self.build_counts_and_probabilities(), f) 

    def save_vocab(self) :
        """Save vocabulary to vocab.json"""
        with open(self.vocab_path, "w") as f:
            json.dump(self.build_vocab(), f)

    def load(self) :
        """Load model.json and vocab.json into the instance. Called once in main() before passing the model to Predictor"""
        self.model = json.load(open(self.model_path, "r", encoding="utf-8"))
        self.vocab = json.load(open(self.vocab_path, "r", encoding="utf-8"))
    
    def main(self) :
        self.save_vocab() # includes building vocab
        self.save_model() # includes building counts and probabilities
        self.load()

def main() :
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../../config/.env")
    print(os.getenv("VOCAB_PATH"))
    ngram_model1 = NGramModel(
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




if __name__ == "__main__":
    main()
        