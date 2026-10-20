
from pathlib import Path
import html
import re

import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "resume_classifier.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


# Load the saved model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_resume(text):
    """Clean resume text using the project's existing cleaning logic."""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_resume(text):
    """Predict the category of a resume."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please provide non-empty resume text.")

    cleaned_text = clean_resume(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]

    return prediction



if __name__ == "__main__":
    test_resumes = {
        "Software Developer": """
            Software developer experienced in Python, Java, JavaScript,
            SQL, web applications and REST API development.
        """,
        "Data Scientist": """
            Data scientist experienced in machine learning, pandas,
            data analysis, statistical modeling and predictive models.
        """,
        "Mechanical Engineer": """
            Mechanical engineer experienced in CAD design, manufacturing,
            mechanical systems, maintenance and quality control.
        """
    }

    for name, resume_text in test_resumes.items():
        category = predict_resume(resume_text)
        print(f"{name} -> Predicted category: {category}")
