"""
Semantic Text Matching Module for SRM CampusFind.
Computes text embeddings using Transformer models (Sentence-BERT / MPNet)
with fallback to Hybrid N-Gram TF-IDF + Synonym Context Mapper & Cosine Similarity.
Provides baselines: Keyword Match, TF-IDF Cosine, and Transformer Embedding Cosine.
"""

import numpy as np
import re
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SYNONYM_CLUSTERS = [
    {"earbuds", "earphones", "airpods", "buds", "headset", "headphones", "earpiece"},
    {"bag", "backpack", "rucksack", "purse", "pouch", "duffel"},
    {"phone", "iphone", "mobile", "smartphone", "cellphone"},
    {"laptop", "notebook", "macbook", "computer"},
    {"id", "identity", "card", "hallticket", "license"},
    {"dark", "black", "navy"},
    {"light", "white", "silver"},
    {"misplaced", "lost", "dropped", "left"},
    {"found", "picked", "discovered"}
]


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Cosine Similarity between two 1D or 2D feature vectors.
    Formula: Cosine Similarity = (A . B) / (||A|| * ||B||)
    """
    v1 = np.array(vec1).flatten()
    v2 = np.array(vec2).flatten()
    
    if len(v1) != len(v2):
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class KeywordBaselineMatcher:
    """Baseline 1: Keyword / Exact Substring Matching."""
    
    @staticmethod
    def match(text1: str, text2: str) -> float:
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)


class TfidfBaselineMatcher:
    """Baseline 2: TF-IDF Vectorizer + Cosine Similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.is_fitted = False
        
    def fit(self, corpus: List[str]):
        if corpus:
            self.vectorizer.fit(corpus)
            self.is_fitted = True
            
    def compute_similarity(self, text1: str, text2: str) -> float:
        if not self.is_fitted:
            matrix = self.vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(matrix[0], matrix[1])[0][0])
        else:
            v1 = self.vectorizer.transform([text1])
            v2 = self.vectorizer.transform([text2])
            return float(cosine_similarity(v1, v2)[0][0])


class TransformerTextEncoder:
    """
    Objective 1 Transformer Semantic Text Encoder.
    Generates dense semantic vector embeddings using Sentence Transformers
    (all-MiniLM-L6-v2) or hybrid semantic projection fallback.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.use_fallback = not HAS_SENTENCE_TRANSFORMERS
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[TransformerTextEncoder] Loaded Transformer model '{model_name}'")
            except Exception as e:
                print(f"[TransformerTextEncoder] Warning: ({e}). Using Hybrid Semantic fallback.")
                self.use_fallback = True
        else:
            print("[TransformerTextEncoder] sentence_transformers library not detected. Using Hybrid Semantic fallback.")
            
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
        self.is_fitted = False
        
    def fit_fallback(self, corpus: List[str]):
        if corpus:
            self.tfidf.fit(corpus)
            self.is_fitted = True

    def _apply_synonym_expansion(self, text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        expanded = list(words)
        for w in words:
            for cluster in SYNONYM_CLUSTERS:
                if w in cluster:
                    expanded.extend(list(cluster))
        return " ".join(expanded)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text or list of texts into dense vectors."""
        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = texts
            
        if not self.use_fallback and self.model is not None:
            embeddings = self.model.encode(input_list, convert_to_numpy=True)
            return embeddings
        else:
            expanded_inputs = [self._apply_synonym_expansion(t) for t in input_list]
            if not self.is_fitted:
                matrix = self.tfidf.fit_transform(expanded_inputs).toarray()
            else:
                matrix = self.tfidf.transform(expanded_inputs).toarray()
                
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return matrix / norms

    def predict_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic text similarity score between two descriptions."""
        e1 = self.encode(text1)
        e2 = self.encode(text2)
        return compute_cosine_similarity(e1[0], e2[0])


if __name__ == "__main__":
    encoder = TransformerTextEncoder()
    t1 = "I misplaced my black wireless earbuds near the library."
    t2 = "Found a pair of dark-colored Bluetooth earphones close to the central library."
    
    kw_score = KeywordBaselineMatcher.match(t1, t2)
    tfidf_score = TfidfBaselineMatcher().compute_similarity(t1, t2)
    transformer_score = encoder.predict_similarity(t1, t2)
    
    print("--- Semantic Text Matching Test ---")
    print(f"Text 1: {t1}")
    print(f"Text 2: {t2}")
    print(f"Keyword Baseline Score:    {kw_score:.4f}")
    print(f"TF-IDF Baseline Score:     {tfidf_score:.4f}")
    print(f"Transformer Semantic Score: {transformer_score:.4f}")
