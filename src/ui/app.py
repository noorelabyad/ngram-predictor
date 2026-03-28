import streamlit as st
import os
import sys
sys.path.append(os.getcwd())
from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


class PredictorUI:
    """Streamlit UI for next-word prediction using the trained N-gram model."""

    def __init__(self, env_path: str):
        load_dotenv(env_path)

        self.normalizer = Normalizer(
            folder_path=os.getenv("TRAIN_RAW_DIR"),
            output_file=os.getenv("TRAIN_TOKENS")
        )

        self.ngram_model = NGramModel(
            token_file=os.getenv("TRAIN_TOKENS"),
            model_path=os.getenv("MODEL_PATH"),
            vocab_path=os.getenv("VOCAB_PATH"),
            unk_threshold=int(os.getenv("UNK_THRESHOLD")),
            ngram_order=int(os.getenv("NGRAM_ORDER"))
        )
        self.ngram_model.load()

        self.predictor = Predictor(
            ngram_model=self.ngram_model,
            normalizer=self.normalizer,
            top_k=int(os.getenv("TOP_K"))
        )

    def run(self):
        """Run the Streamlit UI."""
        st.set_page_config(page_title="Next Word Predictor", layout="centered")

        st.title("📝 Next‑Word Prediction (N‑Gram Model)")
        st.write(
            "Type a sentence and the model will suggest the most likely next words "
            "based on the trained n‑gram language model."
        )

        user_input = st.text_input("Input text", placeholder="Enter a sentence...")

        if user_input:
            predictions = self.predictor.predict_next(user_input)

            st.subheader("Predicted next words:")
            for i, word in enumerate(predictions, 1):
                st.write(f"{i}. `{word}`")

if __name__ == "__main__":
    
    # Choose which env you want to use:
    # env_path = os.path.join(os.getcwd(), "config/.env.test")
    env_path = os.path.join(os.getcwd(), "config/.env")

    ui = PredictorUI(env_path=env_path)
    ui.run()