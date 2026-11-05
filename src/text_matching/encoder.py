"""
Semantic Text Matching Module for SRM CampusFind.
Computes text embeddings using Transformer models (Sentence-BERT / MPNet)
with fallback to Hybrid N-Gram TF-IDF + Synonym Context Mapper & Cosine Similarity.
Provides baselines: Keyword Match, TF-IDF Cosine, and Transformer Embedding Cosine.
"""

import os
import math
import numpy as np
import re
from typing import List, Union

HAS_SENTENCE_TRANSFORMERS = False
if not os.environ.get("VERCEL"):
    try:
        from sentence_transformers import SentenceTransformer
        HAS_SENTENCE_TRANSFORMERS = True
    except Exception:
        HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class SimpleTfidfVectorizer:
    """Lightweight pure-NumPy TF-IDF vectorizer that does not require scipy/sklearn."""
    def __init__(self, stop_words=None, ngram_range=(1, 2)):
        self.stop_words = set(stop_words) if stop_words else set()
        self.ngram_range = ngram_range
        self.vocabulary = {}
        self.idf = {}
        self.is_fitted = False

    def _tokenize(self, text: str) -> List[str]:
        words = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower()) if w not in self.stop_words]
        tokens = list(words)
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            if n > 1 and len(words) >= n:
                for i in range(len(words) - n + 1):
                    tokens.append(" ".join(words[i:i+n]))
        return tokens

    def fit(self, corpus: List[str]):
        from collections import Counter
        doc_count = max(1, len(corpus))
        df = Counter()
        vocab_set = set()
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            vocab_set.update(tokens)
            for t in tokens:
                df[t] += 1
        self.vocabulary = {t: idx for idx, t in enumerate(sorted(vocab_set))}
        self.idf = {t: float(math.log((1.0 + doc_count) / (1.0 + count)) + 1.0) for t, count in df.items()}
        self.is_fitted = True
        return self

    def fit_transform(self, corpus: List[str]):
        self.fit(corpus)
        return self.transform(corpus)

    def transform(self, corpus: List[str]):
        from collections import Counter
        dim = max(1, len(self.vocabulary))
        matrix = np.zeros((len(corpus), dim), dtype=np.float32)
        for row_idx, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            counts = Counter(tokens)
            for t, cnt in counts.items():
                if t in self.vocabulary:
                    col_idx = self.vocabulary[t]
                    matrix[row_idx, col_idx] = float(cnt) * self.idf.get(t, 1.0)
            norm = np.linalg.norm(matrix[row_idx])
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix

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
        if HAS_SKLEARN and not os.environ.get("VERCEL"):
            self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.vectorizer = SimpleTfidfVectorizer()
        self.is_fitted = False
        
    def fit(self, corpus: List[str]):
        if corpus:
            self.vectorizer.fit(corpus)
            self.is_fitted = True
            
    def compute_similarity(self, text1: str, text2: str) -> float:
        if not self.is_fitted:
            res = self.vectorizer.fit_transform([text1, text2])
            matrix = res.toarray() if hasattr(res, "toarray") else res
            return compute_cosine_similarity(matrix[0], matrix[1])
        else:
            r1 = self.vectorizer.transform([text1])
            r2 = self.vectorizer.transform([text2])
            v1 = r1.toarray() if hasattr(r1, "toarray") else r1
            v2 = r2.toarray() if hasattr(r2, "toarray") else r2
            return compute_cosine_similarity(v1[0], v2[0])


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
        
        if HAS_SENTENCE_TRANSFORMERS and not os.environ.get("VERCEL"):
            try:
                self.model = SentenceTransformer(model_name)
                print(f"[TransformerTextEncoder] Loaded Transformer model '{model_name}'")
            except Exception as e:
                print(f"[TransformerTextEncoder] Warning: ({e}). Using Hybrid Semantic fallback.")
                self.use_fallback = True
        else:
            self.use_fallback = True
            
        if HAS_SKLEARN and not os.environ.get("VERCEL"):
            self.tfidf = TfidfVectorizer(ngram_range=(1, 3), analyzer='word')
        else:
            self.tfidf = SimpleTfidfVectorizer(ngram_range=(1, 2))
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
                res = self.tfidf.fit_transform(expanded_inputs)
            else:
                res = self.tfidf.transform(expanded_inputs)
                
            matrix = res.toarray() if hasattr(res, "toarray") else res
                
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
