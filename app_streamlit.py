import streamlit as st
import os
import tempfile
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.file_operations import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
)

# -----------------------
# Helper Functions
# -----------------------

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

def render_highlighted_html(text, sentence_scores, threshold=0.55):
    sentences = split_into_sentences(text)
    parts = []
    for s in sentences:
        score = sentence_scores.get(s, 0)
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
    return extractor(path) if extractor else ""

def compute_sentence_similarity(uploaded_sentences, target_sentences):
    all_sentences = uploaded_sentences + target_sentences
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
    tfidf = vectorizer.fit_transform(all_sentences)
    
    m = len(uploaded_sentences)
    uploaded_mat = tfidf[:m]
    target_mat = tfidf[m:]
    cosine_matrix = cosine_similarity(uploaded_mat, target_mat)
    
    best_uploaded_matches = {s: float(cosine_matrix[i].max()) for i, s in enumerate(uploaded_sentences)}
    best_target_matches = {s: float(cosine_matrix[:, j].max()) for j, s in enumerate(target_sentences)}
    
    return best_uploaded_matches, best_target_matches

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Document Similarity Checker", layout="wide")
st.title("📄 Document Plagiarism Checker")

# Sidebar Settings
st.sidebar.header("Settings")
doc_type = st.sidebar.radio("Select document type:", ["PDF", "DOCX", "PPTX"])
default_dir = r"C:\Users\Benson\Documents\plagiarism_checker_final\sample_pdfs"
directory = st.sidebar.text_input("Folder path to scan:", value=default_dir)
sentence_threshold = st.sidebar.slider("Sentence match threshold", 0.30, 0.85, 0.55, 0.01)
top_n = st.sidebar.number_input("Show top N matches", min_value=1, max_value=5, value=3, step=1)

# -----------------------
# Scan Folder
# -----------------------

def scan_folder():
    if not os.path.exists(directory):
        return []
    files = list_files_by_type(directory, doc_type)
    documents = [{"path": f, "text": extract_text(f, doc_type)} for f in files]
    return [d for d in documents if d["text"].strip()]

if "doc_data" not in st.session_state:
    st.session_state["doc_data"] = scan_folder()

if st.sidebar.button("Scan folder"):
    with st.spinner(f"Scanning {doc_type} files recursively..."):
        st.session_state["doc_data"] = scan_folder()
    st.sidebar.success(f"Scanned {len(st.session_state['doc_data'])} {doc_type} documents with text.")

doc_data = st.session_state["doc_data"]
st.write(f"Scanned {len(doc_data)} {doc_type} documents with text.")

# -----------------------
# Upload File
# -----------------------

st.sidebar.header(f"Upload a {doc_type} to check")
uploaded_file = st.sidebar.file_uploader(f"Upload {doc_type} file", type=[doc_type.lower()])
save_to_folder = st.sidebar.checkbox(f"Save uploaded {doc_type} to folder?")

uploaded_text = ""
if uploaded_file:
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success(f"Saved uploaded file temporarily to {tmp_path}")
    uploaded_text = extract_text(tmp_path, doc_type)

    if save_to_folder:
        dest_path = os.path.join(directory, uploaded_file.name)
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded file saved to folder: {dest_path}")
        st.session_state["doc_data"] = scan_folder()
        doc_data = st.session_state["doc_data"]

# -----------------------
# Comparison
# -----------------------

if uploaded_text.strip() and doc_data:
    st.header("🔗 Document-level similarity (uploaded → folder)")
    
    corpus_texts = [d["text"] for d in doc_data]
    corpus_names = [os.path.basename(d["path"]) for d in doc_data]
    
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
    docs = [uploaded_text] + corpus_texts
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        st.error("Not enough text to compare. Check your documents.")
        st.stop()
    
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df_scores = pd.DataFrame({
        "filename": corpus_names,
        "similarity": cosine_scores
    }).sort_values("similarity", ascending=False).reset_index(drop=True)
    
    st.dataframe(df_scores.head(top_n).style.format({"similarity": "{:.4f}"}))
    
    for idx in range(min(top_n, len(df_scores))):
        top_name = df_scores.loc[idx, "filename"]
        top_score = df_scores.loc[idx, "similarity"]
        st.subheader(f"Top {idx+1} match: {top_name} — score {top_score:.3f}")

        top_text = next(d["text"] for d in doc_data if os.path.basename(d["path"]) == top_name)
        uploaded_sentences = split_into_sentences(uploaded_text)
        top_sentences = split_into_sentences(top_text)

        if uploaded_sentences and top_sentences:
            best_uploaded, best_top = compute_sentence_similarity(uploaded_sentences, top_sentences)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Uploaded Document**")
                st.markdown(render_highlighted_html(uploaded_text, best_uploaded, sentence_threshold), unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{top_name} (Folder)**")
                st.markdown(render_highlighted_html(top_text, best_top, sentence_threshold), unsafe_allow_html=True)
        else:
            st.info("Not enough sentences for sentence-level comparison.")
else:
    st.info(f"Upload a {doc_type} to compare against scanned folder documents.")
