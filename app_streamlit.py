import streamlit as st
import os
import re
import tempfile
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.file_operations import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)

# -------------------------
# Helper functions
# -------------------------

TEXT_EXTRACTORS = {
    "PDF": extract_text_from_pdf,
    "DOCX": extract_text_from_docx,
    "PPTX": extract_text_from_pptx,
}

FILE_EXTENSIONS = {
    "PDF": ".pdf",
    "DOCX": ".docx",
    "PPTX": ".pptx",
}

def split_into_sentences(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def escape_html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def render_highlighted_html(raw_text, matched_sentence_scores, threshold=0.55):
    sentences = split_into_sentences(raw_text)
    parts = []
    for s in sentences:
        score = matched_sentence_scores.get(s, 0)
        esc = escape_html(s)
        if score >= threshold:
            parts.append(f"<mark>{esc}</mark> <small>({score:.2f})</small>")
        else:
            parts.append(esc)
    return "<br><br>".join(parts)

def list_files_by_type(folder_path, file_type):
    ext = FILE_EXTENSIONS[file_type].lower()
    results = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(ext):
                results.append(os.path.join(root, f))
    return results

def extract_text(path, doc_type):
    extractor = TEXT_EXTRACTORS.get(doc_type)
    if extractor:
        return extractor(path)
    return ""

def scan_folder(folder_path, doc_type):
    os.makedirs(folder_path, exist_ok=True)  # ensure folder exists
    all_files = list_files_by_type(folder_path, doc_type)
    documents = [{"path": f, "text": extract_text(f, doc_type)} for f in all_files]
    documents = [d for d in documents if d["text"].strip()]
    st.session_state["doc_data"] = documents
    return documents

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Document Similarity Checker", layout="wide")
st.title("📄 Document Plagiarism Checker")

# Sidebar
st.sidebar.header("Settings")
doc_type = st.sidebar.radio("Select document type:", ["PDF", "DOCX", "PPTX"])
default_dir = "sample_pdfs"  # use relative path for Streamlit Cloud
directory = st.sidebar.text_input("Folder path to scan:", value=default_dir)
sentence_threshold = st.sidebar.slider("Sentence match threshold", 0.30, 0.85, 0.55, 0.01)
top_n = st.sidebar.number_input("Show top N matches", min_value=1, max_value=5, value=3, step=1)

if st.sidebar.button("Scan folder"):
    with st.spinner(f"Scanning {doc_type} files recursively..."):
        documents = scan_folder(directory, doc_type)
    st.sidebar.success(f"Scanned {len(documents)} {doc_type} documents with text.")

# Auto-load cached scan
if "doc_data" not in st.session_state:
    st.session_state["doc_data"] = scan_folder(directory, doc_type)

doc_data = st.session_state["doc_data"]
st.write(f"Scanned {len(doc_data)} {doc_type} documents with text.")

# Upload file
st.sidebar.header(f"Upload a {doc_type} to check")
uploaded_file = st.sidebar.file_uploader(f"Upload {doc_type} file", type=[doc_type.lower()])
save_to_folder = st.sidebar.checkbox(f"Save uploaded {doc_type} to folder?")

uploaded_text = ""
uploaded_name = None
if uploaded_file:
    uploaded_name = uploaded_file.name
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Saved uploaded file temporarily to {tmp_path}")
    uploaded_text = extract_text(tmp_path, doc_type)

    if save_to_folder:
        os.makedirs(directory, exist_ok=True)
        dest_path = os.path.join(directory, uploaded_file.name)
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded file saved to folder: {dest_path}")
        doc_data = scan_folder(directory, doc_type)
        st.session_state["doc_data"] = doc_data

# Comparison
if uploaded_file and uploaded_text.strip() and len(doc_data) > 0:
    st.header("🔗 Document-level similarity (uploaded → folder)")

    corpus_texts = [d["text"] for d in doc_data]
    corpus_names = [os.path.basename(d["path"]) for d in doc_data]

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
    docs = [uploaded_text] + corpus_texts
    try:
        tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        st.error("Not enough text to compare. Check your documents.")
        st.stop()

    cosine_with_uploaded = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    df_scores = pd.DataFrame({
        "filename": corpus_names,
        "similarity": cosine_with_uploaded
    }).sort_values("similarity", ascending=False).reset_index(drop=True)

    st.dataframe(df_scores.head(top_n).style.format({"similarity": "{:.4f}"}))

    for idx in range(min(top_n, len(df_scores))):
        top_name = df_scores.loc[idx, "filename"]
        top_score = df_scores.loc[idx, "similarity"]
        st.subheader(f"Top {idx+1} match: {top_name} — score {top_score:.3f}")

        top_text = next((d["text"] for d in doc_data if os.path.basename(d["path"]) == top_name), "")

        uploaded_sentences = split_into_sentences(uploaded_text)
        top_sentences = split_into_sentences(top_text)

        if uploaded_sentences and top_sentences:
            sent_docs = uploaded_sentences + top_sentences
            sent_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
            sent_tfidf = sent_vectorizer.fit_transform(sent_docs)

            m = len(uploaded_sentences)
            uploaded_mat = sent_tfidf[:m]
            top_mat = sent_tfidf[m:]

            sent_cosine = cosine_similarity(uploaded_mat, top_mat)

            best_matches = {uploaded_sentences[i]: float(sent_cosine[i].argmax()) for i in range(m)}
            top_doc_scores = {top_sentences[j]: float(sent_cosine[:, j].max()) for j in range(len(top_sentences))}

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Uploaded document**")
                html_left = render_highlighted_html(uploaded_text, best_matches, sentence_threshold)
                st.markdown(html_left, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{top_name} (folder)**")
                html_right = render_highlighted_html(top_text, top_doc_scores, sentence_threshold)
                st.markdown(html_right, unsafe_allow_html=True)
        else:
            st.info("Not enough sentences for sentence-level comparison.")
else:
    st.info(f"Upload a {doc_type} to compare against scanned folder documents.")
