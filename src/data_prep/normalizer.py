import os
from pydoc import text
import re
from tracemalloc import start

class Normalizer:

    """module's responsibility: loading, cleaning, tokenizing, and saving the corpus
    used in In Module 1 (Data Prep) to processes whole raw files, 
    and In Module 3 (Inference), only normalize(text) is called on a single input string 
    to prepare it for lookup
    """
    def __init__(self, folder_path = None, output_file = None) :
        self.folder_path = folder_path
        self.output_file = output_file

    def load (self) :
        """Load all .txt files from a folder"""
        text = ""
        for filename in sorted(os.listdir(self.folder_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"
        return text

    @staticmethod
    def strip_gutenberg(text) :
        """Remove Gutenberg header and footer
        Extract only text in between the lines:
         *** START OF THE PROJECT GUTENBERG
         *** END OF THE PROJECT GUTENBERG, removing intro and references sections"""
        
        start = "*** START OF THE PROJECT GUTENBERG"
        end = "*** END OF THE PROJECT GUTENBERG"
        if start not in text or end not in text:
            return text
        pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return "\n".join(m.strip() for m in matches)

    @staticmethod
    def lowercase(text) :
        """Lowercase all text"""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text) :
        """Remove all punctuation as well as underscores
         retaining fullstops and replacing all excalamations 
         and question marks with fullstops to identify sentences"""

        text = re.sub(r"’", "'", text)
        text = re.sub(r"[^\w\s'.?!:]|_", " ", text)
        return re.sub(r'[?!:]', '.', text)

    @staticmethod
    def remove_numbers(text) :
        """Remove all numbers"""
        return re.sub(r"\d+", "", text)
    
    @staticmethod
    def remove_whitespaces(text) :
        """Remove extra whitespaces and blank lines"""
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(" ".join(stripped.split()))

        return "\n".join(cleaned_lines)

    @staticmethod
    def normalize(text) :
        """Apply all normalization steps in order: 
        lowercase → remove punctuation → remove numbers → remove whitespace. 
        This is the single method that other modules call to normalize text consistently."""
        text = Normalizer.lowercase(text)
        text = Normalizer.remove_punctuation(text)
        text = Normalizer.remove_numbers(text)
        text = Normalizer.remove_whitespaces(text)
        return text
        
    @staticmethod
    def sentence_tokenize(text) :
        """Split text into sentences using fullstops as delimiters"""
        sentences = text.split(".")
        return sentences
    
    @staticmethod
    def word_tokenize(sentence) :
        """Split a single sentence into a list of tokens"""
        tokens = sentence.split()
        return tokens
    

    def save(self, word_tokens) :
        """Save the tokenized words to a file, one sentence per line, with tokens separated by spaces"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            for tokens in word_tokens:
                f.write(" ".join(tokens) + "\n")


    def main(self) :
        text = self.load()
        text = self.strip_gutenberg(text)
        text = self.normalize(text)
        sentences = self.sentence_tokenize(text)
        word_tokens = [self.word_tokenize(sentence) for sentence in sentences]
        self.save(word_tokens)


def main() :
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="../../config/.env")
    normalizer1 = Normalizer(folder_path=os.getenv("TRAIN_RAW_DIR"), output_file=os.getenv("TRAIN_TOKENS"))
    
    # Training
    normalizer1.main()

    # Normalizer Test
    print(Normalizer.normalize("  Hello, World! This is a test. 123   "))



if __name__ == "__main__":
    main()