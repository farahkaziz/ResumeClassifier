
from pathlib import Path
import os
import html
import re
import joblib


# Chemin du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "resume_classifier.pkl"
LOCAL_VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


def load_artifact(filename, local_path):
    """Charge un fichier local ou le télécharge depuis Hugging Face."""

    # Utiliser le fichier local s'il existe
    if local_path.exists():
        return joblib.load(local_path)

    # Lire la configuration des secrets
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")

    try:
        import streamlit as st

        repo_id = repo_id or st.secrets.get("HF_REPO_ID")
        token = token or st.secrets.get("HF_TOKEN")
    except Exception:
        pass

    if not repo_id or not token:
        raise RuntimeError(
            "Configuration Hugging Face manquante. "
            "Vérifiez HF_REPO_ID et HF_TOKEN dans les secrets."
        )

    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )

    return joblib.load(file_path)


# Charger le modèle et le vectoriseur
model = load_artifact(
    "resume_classifier.pkl",
    LOCAL_MODEL_PATH,
)

vectorizer = load_artifact(
    "tfidf_vectorizer.pkl",
    LOCAL_VECTORIZER_PATH,
)


def clean_resume(text):
    """Nettoie le texte du CV."""

    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_resume(text):
    """Prédit la catégorie professionnelle du CV."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide non-empty resume text.")

    cleaned_text = clean_resume(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]

    return prediction
