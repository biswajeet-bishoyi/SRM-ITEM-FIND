"""
Vector Database & Retrieval System for SRM CampusFind.
Provides fast candidate vector indexing, cosine similarity top-K search,
and multimodal record lookup.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class VectorDatabase:
    """
    Vector Index and Database for lost-and-found report embeddings.
    Stores dense text embeddings, image embeddings, and associated record metadata.
    """
    
    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.text_embeddings: List[np.ndarray] = []
        self.image_embeddings: List[Optional[np.ndarray]] = []

    def clear(self):
        self.records = []
        self.text_embeddings = []
        self.image_embeddings = []

    def add_record(
        self, 
        record: Dict[str, Any], 
        text_emb: np.ndarray, 
        image_emb: Optional[np.ndarray]
    ):
        """Add a lost/found report record with its text and image embeddings (image_emb is None if no photo)."""
        self.records.append(record)
        
        # Ensure vectors are 1D normalized numpy arrays
        t_vec = np.array(text_emb).flatten()
        t_norm = np.linalg.norm(t_vec)
        if t_norm > 0:
            t_vec = t_vec / t_norm
            
        i_vec = self._normalize(image_emb)
            
        self.text_embeddings.append(t_vec)
        self.image_embeddings.append(i_vec)

    @staticmethod
    def _normalize(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vec is None:
            return None
        v = np.array(vec).flatten()
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def search(
        self, 
        query_text_emb: np.ndarray, 
        query_image_emb: Optional[np.ndarray], 
        target_status: str = "FOUND", 
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float, float]]:
        """
        Search top-K nearest records matching target status (e.g. searching LOST queries against FOUND records).
        Returns list of tuples: (record, text_sim, image_sim).
        image_sim is None when either the query or the record has no photo.
        """
        if not self.records:
            return []
            
        q_text = np.array(query_text_emb).flatten()
        q_t_norm = np.linalg.norm(q_text)
        if q_t_norm > 0:
            q_text = q_text / q_t_norm
            
        q_image = self._normalize(query_image_emb)
            
        candidate_results = []
        
        for idx, rec in enumerate(self.records):
            # Only compare against opposing status (LOST query matches FOUND database items, or vice versa)
            if target_status and rec.get("status") != target_status:
                continue
                
            t_vec = self.text_embeddings[idx]
            i_vec = self.image_embeddings[idx]
            
            # Vector Dot Product (Cosine Similarity since vectors are normalized)
            text_sim = float(np.dot(q_text, t_vec)) if len(q_text) == len(t_vec) else 0.5
            image_sim = None
            if q_image is not None and i_vec is not None and len(q_image) == len(i_vec):
                image_sim = max(0.0, min(1.0, float(np.dot(q_image, i_vec))))
            
            # Clip between [0, 1]
            text_sim = max(0.0, min(1.0, text_sim))
            
            candidate_results.append((rec, text_sim, image_sim))
            
        # Sort by average text + image similarity (text only when no photo is available)
        candidate_results.sort(key=lambda x: x[1] if x[2] is None else (x[1] * 0.5 + x[2] * 0.5), reverse=True)
        return candidate_results[:top_k]


if __name__ == "__main__":
    vdb = VectorDatabase()
    rec1 = {"id": "FOUND_100", "status": "FOUND", "description": "Found dark earbuds"}
    t_emb1 = np.random.randn(384)
    i_emb1 = np.random.randn(512)
    vdb.add_record(rec1, t_emb1, i_emb1)
    
    results = vdb.search(t_emb1, i_emb1, target_status="FOUND", top_k=1)
    print("--- Vector DB Retrieval Test ---")
    print(f"Retrieved {len(results)} items. Best match ID: {results[0][0]['id']}")
