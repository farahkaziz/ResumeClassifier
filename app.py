import pickle
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError


st.set_page_config(page_title="Resume Classifier", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    :root {
        --navy: #0B162C;
        --slate: #1C2942;
        --blue: #3B556D;
        --teal: #5FC2BA;
        --white: #FFFFFF;
    }
    .stApp {
        background-color: var(--navy);
        color: var(--white);
    }
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid rgba(95, 194, 186, 0.25);
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: #5FC2BA;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero h1 {
        color: #FFFFFF;
        font-size: 2.6rem;
        margin: 0;
    }
    .hero p {
        color: #C4CFDC;
        font-size: 1.05rem;
        margin: 0.65rem 0 0;
        max-width: 660px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1C2942;
        border-color: rgba(95, 194, 186, 0.22);
        border-radius: 14px;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stFileUploader"] section {
        background-color: #0B162C;
        border-color: #3B556D;
        color: #FFFFFF;
    }
    div.stButton > button[kind="primary"] {
        background-color: #5FC2BA;
        border: 1px solid #5FC2BA;
        color: #0B162C;
        font-weight: 700;
        min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #79D0C8;
        border-color: #79D0C8;
        color: #0B162C;
    }
    .result-label {
        color: #C4CFDC;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .result-category {
        color: #5FC2BA;
        font-size: 1.8rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .app-footer {
        border-top: 1px solid rgba(95, 194, 186, 0.2);
        color: #9BAABD;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Analyse de CV</div>
        <h1>Resume Classifier</h1>
        <p>Collez le texte d’un CV ou importez un PDF pour obtenir une catégorie
        professionnelle estimée par le modèle.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

resume_text = ""
uploaded_pdf = None

with st.container(border=True):
    st.subheader("Fournir un CV")
    input_method = st.radio(
        "Choisissez une méthode",
        ("Coller le texte du CV", "Importer un fichier PDF"),
        horizontal=True,
    )

    if input_method == "Coller le texte du CV":
        resume_text = st.text_area("Contenu du CV", height=280)
    else:
        uploaded_pdf = st.file_uploader("Choisir un fichier PDF", type=["pdf"])

if st.button("Analyser le CV", type="primary", use_container_width=True):
    if input_method == "Importer un fichier PDF":
        if uploaded_pdf is None:
            st.warning("Veuillez importer un fichier PDF avant de lancer l’analyse.")
        else:
            try:
                reader = PdfReader(BytesIO(uploaded_pdf.getvalue()))
                resume_text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except (
                KeyError,
                OSError,
                PdfReadError,
                PyPdfError,
                TypeError,
                UnicodeError,
                ValueError,
            ) as error:
                st.error(f"Impossible de lire ou d’extraire le texte du PDF : {error}")
                resume_text = None

    if resume_text is not None:
        if not resume_text.strip():
            if input_method == "Importer un fichier PDF" and uploaded_pdf is not None:
                st.warning(
                    "Aucun texte n’a pu être extrait du PDF. "
                    "Il s’agit peut-être d’un document scanné."
                )
            else:
                st.warning("Veuillez coller le contenu d’un CV avant de lancer l’analyse.")
        else:
            try:
                from src.predict import predict_resume

                category = predict_resume(resume_text)
            except (
                AttributeError,
                EOFError,
                ImportError,
                OSError,
                pickle.UnpicklingError,
                ValueError,
            ) as error:
                st.error(f"Impossible d’analyser le CV : {error}")
            else:
                with st.container(border=True):
                    st.markdown(
                        '<div class="result-label">Catégorie prédite</div>'
                        f'<div class="result-category">{category}</div>',
                        unsafe_allow_html=True,
                    )
                    st.warning(
                        "Cette prédiction est indicative et peut être incorrecte."
                    )

st.markdown(
    '<div class="app-footer">Propulsé par Streamlit, scikit-learn et pypdf</div>',
    unsafe_allow_html=True,
)
