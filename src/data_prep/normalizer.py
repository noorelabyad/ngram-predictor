import os
import sys
sys.path.append(os.getcwd())
import re
import nltk
from nltk.tokenize import sent_tokenize


class Normalizer:

    """module's responsibility: loading raw text files, cleaning and normalizing the text, and saving the processed tokens to a file for model training.
    The normalization steps are:
    1. Strip Gutenberg header and footer
    2. Lowercase all text
    3. Remove punctuation
    4. Remove numbers
    5. Remove extra whitespaces and blank lines
    The same normalization steps are applied to user input in inference to ensure consistency between training and inference.
    Class Attributes:
    folder_path: path to the folder containing raw text files to be processed
    output_file: path to the file where processed tokens will be saved"""
    
    def __init__(self, folder_path = None, output_file = None) :
        self.folder_path = folder_path
        self.output_file = output_file

    def load (self) :
        """Load all .txt files from the folder_path attribute of the instance and concatenate them into a single string and return it."""
        text = ""
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"
        return text

    @staticmethod
    def strip_gutenberg(text):
        """
        Extract only the lines in between the Gutenberg header and footer
        from the input string (parameter 'text') that may contain several
        concatenated Gutenberg texts, and return the cleaned string.
        """
        start = "*** START OF THE PROJECT GUTENBERG"
        end = "*** END OF THE PROJECT GUTENBERG"

        has_start = start in text
        has_end = end in text

        # Case 1: neither exists
        if not has_start and not has_end:
            return text

        # both exist (possibly multiple times)
        if has_start and has_end:
            pattern = (
                        re.escape(start) + r"[^\n]*\n" # start + till end of line
                        r"(.*?)" +
                        re.escape(end)
                    )
            matches = re.findall(pattern, text, flags=re.DOTALL)
            return "\n".join(m.strip() for m in matches)

        # only START exists: remove everything before and including it
        if has_start:
            return re.split(re.escape(start), text, maxsplit=1)[1].lstrip()

        # only END exists: remove everything after and including it
        if has_end:
            return re.split(re.escape(end), text, maxsplit=1)[0].rstrip()


    @staticmethod
    def lowercase(text) :
        """Lowercase all text from an input string (parameter 'text') and return the cleaned string"""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text) :
        """given a parameter input string (parameter 'text'); removes all punctuation and return the cleaned text"""

        text = re.sub(r"’", "'", text)
        text = re.sub(r"[^\w\s]|_", " ", text)
        return text

    @staticmethod
    def remove_numbers(text) :
        """Remove all numbers from an input string (parameter 'text') and return the cleaned string"""
        return re.sub(r"\d+", "", text)
    
    @staticmethod
    def remove_whitespaces(text) :
        """Remove extra whitespaces and blank lines from an input string (parameter 'text') and return the cleaned string"""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(" ".join(stripped.split()))

        return "\n".join(cleaned_lines)

    @staticmethod
    def normalize(text) :
        """ for a given input string (parameter 'text'), Apply all normalization steps in order: 
        lowercase → remove punctuation → remove numbers → remove whitespace. Return the cleaned string."""
        text = Normalizer.lowercase(text)
        text = Normalizer.remove_punctuation(text)
        text = Normalizer.remove_numbers(text)
        text = Normalizer.remove_whitespaces(text)
        return text
        
    @staticmethod

    def sentence_tokenize(text):
        """
        Sentence tokenize using NLTK.
        Automatically downloads required resources if missing.
        """
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")

        return sent_tokenize(text)



    @staticmethod
    def word_tokenize(sentence) :
        """for an input string (parameter 'sentence'), Split a single sentence into a list of tokens and return the list."""
        tokens = sentence.split()
        return tokens
    

    def save(self, word_tokens):
        """
        Save the tokenized words (input parameter 'word_tokens') to the output_file
        attribute of the instance, one sentence per line, with tokens separated by spaces.
        Creates directories if they do not exist.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            for tokens in word_tokens:
                f.write(" ".join(tokens) + "\n")


    def main(self):
        """Run the whole pipeline: load raw text, clean and tokenize it, and save the processed tokens to a file."""
        text = self.load()
        text = Normalizer.strip_gutenberg(text)
        sentences = Normalizer.sentence_tokenize(text)
        sentences = [Normalizer.normalize(sentence) for sentence in sentences]
        word_tokens = [Normalizer.word_tokenize(sentence) for sentence in sentences]

        self.save(word_tokens)


def main() :
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))
    normalizer1 = Normalizer(folder_path=os.getenv("TRAIN_RAW_DIR"), output_file=os.getenv("TRAIN_TOKENS"))
    
    # Training
    normalizer1.main()

    # Normalizer Test
    print(Normalizer.normalize("  \"Hello. Mr. World\" This is a test, 123   "))



if __name__ == "__main__":
    main()