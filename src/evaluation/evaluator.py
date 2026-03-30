import math
import os


class Evaluator:
    """
    Computes cross-entropy and perplexity on a held-out evaluation corpus.
    Assumes evaluation text has already been normalized and saved to EVAL_TOKENS.
    """

    def __init__(self, model, normalizer, eval_file):
        """
        model: pre-loaded NGramModel
        normalizer: Normalizer instance (kept for symmetry)
        eval_file: path to tokenized evaluation corpus (EVAL_TOKENS)
        """
        self.model = model
        self.normalizer = normalizer
        self.eval_file = eval_file

    def score_word(self, word, context_tokens):
        """
        Return log2 P(word | context) using NGramModel.lookup().
        context_tokens: list of previous tokens
        Returns None if word has zero probability.
        """
        context = " ".join(context_tokens)

        probs = self.model.lookup(context)   # returns {word: prob}
        # print(f"DEBUG: Context='{context}' | Word='{word}' | Probs={probs}")  # Debug log 

        if word not in probs or probs[word] <= 0.0:
            return None

        return math.log2(probs[word])

    def _load_eval_tokens(self):
        """
        Load tokenized evaluation sentences from EVAL_TOKENS.
        Returns a flat list of tokens.
        """
        if not os.path.isfile(self.eval_file):
            raise FileNotFoundError(f"Eval tokens file not found: {self.eval_file}")

        tokens = []

        with open(self.eval_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens.extend(line.split())

        return tokens

    def compute_perplexity(self):
        """
        Compute perplexity over the evaluation corpus.
        Prints a warning if more than 20% of words are skipped.
        """
        tokens = self._load_eval_tokens()

        total_log_prob = 0.0
        evaluated = 0
        skipped = 0

        n = self.model.ngram_order

        for i in range(n - 1, len(tokens)):
            context_tokens = tokens[i - n + 1 : i]
            word = tokens[i]

            log_prob = self.score_word(word, context_tokens)

            if log_prob is None:
                skipped += 1
                continue

            total_log_prob += log_prob
            evaluated += 1

        if evaluated == 0:
            raise ValueError("No words evaluated; perplexity undefined.")

        cross_entropy = - total_log_prob / evaluated
        perplexity = 2 ** cross_entropy

        skip_ratio = skipped / (evaluated + skipped)
        if skip_ratio > 0.2:
            print(
                f"WARNING: {skip_ratio:.1%} of words were skipped due to zero probability."
            )

        return perplexity, evaluated, skipped

    def run(self):
        """
        Run evaluation and print results.
        """
        ppl, evaluated, skipped = self.compute_perplexity()

        print(f"Perplexity: {ppl:.2f}")
        print(f"Words evaluated: {evaluated:,}")
        print(f"Words skipped (zero probability): {skipped:,}")