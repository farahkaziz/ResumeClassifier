# Resume Classifier

A machine learning project that classifies resumes into professional categories using their text. The project includes a Streamlit application for entering resume text or uploading a PDF, and uses saved model artifacts for prediction.

## Features

- Predicts a resume category from pasted text.
- Accepts PDF uploads and extracts text for prediction. Scanned PDFs without extractable text are not supported.
- Provides a Streamlit user interface.
- Displays predictions as indicative; they may be incorrect.

## Technologies

- Python
- pandas for working with the dataset
- scikit-learn for TF-IDF features and the LinearSVC classifier
- joblib for loading and saving the model artifacts
- Streamlit for the web interface
- pypdf for extracting text from PDF files
- Jupyter Notebook and Matplotlib for exploration and evaluation

## Dataset

The project uses the Resume Dataset stored in `data/raw/Resume.csv`. Its processed version, `data/processed/Resume_clean.csv`, contains **2,481 resumes** across **24 categories**.

The dataset does not include a dedicated Data Science category. Data Science-related resumes must therefore be assigned to one of the categories present in the dataset.

## Model

The final model documented in the notebook is a `LinearSVC` classifier with `C=2` and balanced class weights. Resume text is represented with a TF-IDF vectorizer configured to ignore common English stop words and terms appearing in fewer than two training documents.

The trained classifier and fitted vectorizer are stored in:

- `models/resume_classifier.pkl`
- `models/tfidf_vectorizer.pkl`

## Evaluation

The notebook reports results on the held-out test set of **373 resumes**, which was not used to fit the final model or vectorizer:

| Metric | Test result |
| --- | ---: |
| Accuracy | 0.6997 (about 70%) |
| Macro F1-score | 0.6726 (about 0.67) |
| Weighted F1-score | 0.6904 |

These are the results recorded for the existing test evaluation; they are not a guarantee of performance on new resumes.

## Project structure

```text
ML_cv_classification/
├── app.py
├── data/
│   ├── processed/
│   │   └── Resume_clean.csv
│   └── raw/
│       └── Resume.csv
├── models/
│   ├── resume_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── clean.py
│   └── predict.py
└── requirements.txt
```

## Installation and launch

From the project root, create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install streamlit pypdf joblib pandas scikit-learn
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install streamlit pypdf joblib pandas scikit-learn
```

Start the application from the project root:

```bash
streamlit run app.py
```

The pinned application dependencies are listed in `requirements.txt` and can be installed with `python -m pip install -r requirements.txt`.

## Limitations

- Predictions are indicative and can be wrong; the model can confuse related categories.
- The available labels are limited to the 24 categories in the dataset, which has no dedicated Data Science category.
- PDF support depends on text being extractable. Scanned PDFs are not processed with OCR.
- The test metrics describe the existing held-out test set and may not represent results on new or differently formatted resumes.
