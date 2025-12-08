import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.chunk import RegexpParser
from sentence_transformers import SentenceTransformer


class TextProcessor:
    def __init__(self):
        # Load the embedding model for embedding-based similarity
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # --- Preprocessing Methods ---
    @staticmethod
    def preprocess_text(text, cleaned=True):
        """
        Preprocess text for general similarity calculations.
        - Lowercase the text.
        - Optionally clean by removing punctuation and stop words, and applying lemmatization.
        """
        text = text.lower()
        if cleaned:
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)
        if cleaned:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    @staticmethod
    def preprocess_text_for_spelling(text):
        """
        Preprocess text specifically for spelling similarity.
        - Lowercase the text.
        - Remove punctuation and stop words (no lemmatization applied).
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    # --- Noun Phrase Extraction ---
    @staticmethod
    def extract_noun_phrases(tokens):
        """
        Extract noun phrases from tokenized text using POS tagging and chunking.
        """
        pos_tags = pos_tag(tokens)  # Part-of-speech tagging
        grammar = "NP: {<DT>?<JJ>*<NN.*>+}"  # Define grammar for noun phrases
        chunk_parser = RegexpParser(grammar)
        tree = chunk_parser.parse(pos_tags)

        noun_phrases = []
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            phrase = " ".join([word for word, _ in subtree.leaves()])
            noun_phrases.append(phrase)

        return noun_phrases

    # --- Similarity Methods ---
    @staticmethod
    def jaccard_similarity(tokens1, tokens2):
        """
        Compute Jaccard similarity between two sets of tokens.
        """
        intersection = len(set(tokens1).intersection(tokens2))
        union = len(set(tokens1).union(tokens2))
        return intersection / union if union else 0

    @staticmethod
    def compute_cosine_similarity(text1, text2):
        """
        Compute cosine similarity between two sets of tokens using TF-IDF.
        """
        combined_text = [' '.join(text1), ' '.join(text2)]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(combined_text)
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def compute_embedding_similarity(self, text1, text2):
        """
        Compute similarity between two texts using embeddings after cleaning the text.
        """
        # Clean the texts first
        cleaned_text1 = self.preprocess_text(text1, cleaned=True)  # Ensure preprocessing happens here
        cleaned_text2 = self.preprocess_text(text2, cleaned=True)

        # Convert cleaned tokens to a single string
        text1_clean = ' '.join(cleaned_text1)
        text2_clean = ' '.join(cleaned_text2)

        # Generate embeddings
        embedding1 = self.embedding_model.encode(text1_clean)
        embedding2 = self.embedding_model.encode(text2_clean)

        # Compute cosine similarity between embeddings
        return cosine_similarity([embedding1], [embedding2])[0][0]


    # --- Cleaned and Uncleaned Cosine Similarity ---
    def cosine_similarity_cleaned(self, text1, text2):
        """
        Compute cosine similarity for cleaned text.
        """
        tokens1_cleaned = self.preprocess_text(text1, cleaned=True)
        tokens2_cleaned = self.preprocess_text(text2, cleaned=True)
        return self.compute_cosine_similarity(tokens1_cleaned, tokens2_cleaned)

    def cosine_similarity_uncleaned(self, text1, text2):
        """
        Compute cosine similarity for uncleaned text.
        """
        tokens1_uncleaned = self.preprocess_text(text1, cleaned=False)
        tokens2_uncleaned = self.preprocess_text(text2, cleaned=False)
        return self.compute_cosine_similarity(tokens1_uncleaned, tokens2_uncleaned)

    # --- Phrase-Based Cosine Similarity ---
    def cosine_similarity_phrases_cleaned(self, text1, text2):
        """
        Compute cosine similarity for cleaned noun phrases.
        """
        tokens1_cleaned = self.preprocess_text(text1, cleaned=True)
        tokens2_cleaned = self.preprocess_text(text2, cleaned=True)
        phrases1_cleaned = self.extract_noun_phrases(tokens1_cleaned)
        phrases2_cleaned = self.extract_noun_phrases(tokens2_cleaned)
        return self.compute_cosine_similarity(phrases1_cleaned, phrases2_cleaned)

    def cosine_similarity_phrases_uncleaned(self, text1, text2):
        """
        Compute cosine similarity for uncleaned noun phrases.
        """
        tokens1_uncleaned = self.preprocess_text(text1, cleaned=False)
        tokens2_uncleaned = self.preprocess_text(text2, cleaned=False)
        phrases1_uncleaned = self.extract_noun_phrases(tokens1_uncleaned)
        phrases2_uncleaned = self.extract_noun_phrases(tokens2_uncleaned)
        return self.compute_cosine_similarity(phrases1_uncleaned, phrases2_uncleaned)

    # --- Jaccard Similarity for Spelling ---
    def jaccard_similarity_for_spelling(self, text1, text2):
        """
        Compute Jaccard similarity for spelling detection.
        Preprocessing includes stop word removal but excludes lemmatization.
        """
        tokens1 = self.preprocess_text_for_spelling(text1)
        tokens2 = self.preprocess_text_for_spelling(text2)
        return self.jaccard_similarity(tokens1, tokens2)




if __name__ == "__main__":
    directory = r"C:\Users\Benson\Documents\plagiarism_checker_final\sample_pdfs" 
    pdf_data = extract_pdfs_into_list(directory)
    
    print(f"\nTotal PDFs processed: {len(pdf_data)}\n")
    for pdf in pdf_data:
        print(f"📄 PDF Path: {pdf['path']}")
        print(f"📝 Text length: {len(pdf['text'])}")
        print(f"🖼️ Grayscale images extracted: {len(pdf['image'])}")
        print(f"🎨 Color images extracted: {len(pdf['image_color'])}\n")


    



    