"""
Comprehensive Model Evaluation and Baseline Comparison Framework for SRM CampusFind.
Evaluates Precision, Recall, F1-Score, Accuracy, Recall@K, and MRR across baselines.
"""

import numpy as np
from typing import List, Dict, Any

from src.preprocessing.dataset_generator import generate_dataset
from src.text_matching.encoder import KeywordBaselineMatcher, TfidfBaselineMatcher, TransformerTextEncoder, compute_cosine_similarity
from src.image_matching.encoder import ImageEncoder
from src.multimodal.fusion import MultimodalFusionEngine, compute_location_similarity, compute_time_similarity, compute_attribute_similarity
from src.retrieval.vector_db import VectorDatabase


class SystemEvaluator:
    """
    Comprehensive Model Evaluator for SRM CampusFind.
    Runs experimental benchmarks comparing Baselines vs Proposed Multimodal Architecture.
    """
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        text_encoder: TransformerTextEncoder = None,
        image_encoder: ImageEncoder = None
    ):
        # Only curated records carry ground-truth match labels, so user-submitted reports are excluded
        self.dataset = [r for r in dataset if r.get("data_source") == "REAL_CURATED"]
        # Reuse already-loaded encoders when provided, to avoid reloading model weights on every run
        self.text_encoder = text_encoder or TransformerTextEncoder()
        self.image_encoder = image_encoder or ImageEncoder()
        self.fusion_engine = MultimodalFusionEngine()
        self.tfidf_matcher = TfidfBaselineMatcher()
        
        # Fit vectorizers on dataset corpus
        corpus = [r["description"] for r in self.dataset]
        self.tfidf_matcher.fit(corpus)
        self.text_encoder.fit_fallback(corpus)

    def evaluate_all(self) -> Dict[str, Any]:
        """Run full evaluation suite and calculate experimental metrics."""
        
        # Separate Lost queries and Found target records
        lost_items = [r for r in self.dataset if r["status"] == "LOST"]
        found_items = [r for r in self.dataset if r["status"] == "FOUND"]
        
        # Build Vector DB with Found items
        vdb = VectorDatabase()
        found_t_embs = [self.text_encoder.encode(f["description"]) for f in found_items]
        found_i_embs = [self.image_encoder.encode(f["image_filename"]) for f in found_items]
        
        for idx, f in enumerate(found_items):
            vdb.add_record(f, found_t_embs[idx], found_i_embs[idx])

        # Benchmark counters for each model architecture
        models_to_test = [
            "Baseline 1 (Keyword)", 
            "Baseline 2 (TF-IDF)", 
            "Baseline 3 (Transformer Text)", 
            "Baseline 4 (Image Only)", 
            "Proposed Multimodal"
        ]
        results_by_model = {m: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "mrr_sum": 0.0, "r_at_1": 0, "r_at_3": 0, "evaluated": 0} for m in models_to_test}

        def image_similarity(q_emb, f_emb):
            # None when either report has no photo (visual modality missing)
            if q_emb is None or f_emb is None:
                return None
            return compute_cosine_similarity(q_emb, f_emb)

        for lost in lost_items:
            target_match_id = lost["match_target_id"]
            
            # Extract query embeddings
            q_t_emb = self.text_encoder.encode(lost["description"])
            q_i_emb = self.image_encoder.encode(lost["image_filename"])
            
            # Evaluate each baseline and proposed model
            for model_name in models_to_test:
                # The image-only baseline can only be scored when the query has a photo
                if model_name == "Baseline 4 (Image Only)" and q_i_emb is None:
                    continue
                results_by_model[model_name]["evaluated"] += 1
                scored_candidates = []
                
                for found_idx, found in enumerate(found_items):
                    if model_name == "Baseline 1 (Keyword)":
                        score = KeywordBaselineMatcher.match(lost["description"], found["description"])
                    elif model_name == "Baseline 2 (TF-IDF)":
                        score = self.tfidf_matcher.compute_similarity(lost["description"], found["description"])
                    elif model_name == "Baseline 3 (Transformer Text)":
                        score = compute_cosine_similarity(q_t_emb[0], found_t_embs[found_idx][0])
                    elif model_name == "Baseline 4 (Image Only)":
                        score = image_similarity(q_i_emb, found_i_embs[found_idx]) or 0.0
                    elif model_name == "Proposed Multimodal":
                        t_sim = compute_cosine_similarity(q_t_emb[0], found_t_embs[found_idx][0])
                        i_sim = image_similarity(q_i_emb, found_i_embs[found_idx])
                        loc_sim = compute_location_similarity(lost["coordinates"], found["coordinates"])
                        time_sim = compute_time_similarity(lost["timestamp"], found["timestamp"])
                        attr_sim = compute_attribute_similarity(lost, found)
                        
                        f_res = self.fusion_engine.fuse(t_sim, i_sim, loc_sim, time_sim, attr_sim)
                        score = f_res["final_score"]
                        
                    scored_candidates.append((found, max(0.0, min(1.0, score))))
                    
                # Sort candidates by predicted score descending
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Check top-1 match
                top1_item, top1_score = scored_candidates[0]
                is_correct = (top1_item["id"] == target_match_id)
                threshold = 0.50 if "Baseline" in model_name else 0.65
                
                if is_correct and top1_score >= threshold:
                    results_by_model[model_name]["tp"] += 1
                elif not is_correct and top1_score >= threshold:
                    results_by_model[model_name]["fp"] += 1
                elif is_correct and top1_score < threshold:
                    results_by_model[model_name]["fn"] += 1
                else:
                    results_by_model[model_name]["tn"] += 1
                    
                # Recall@K and MRR
                rank = -1
                for r_idx, (cand, _) in enumerate(scored_candidates):
                    if cand["id"] == target_match_id:
                        rank = r_idx + 1
                        break
                        
                if rank == 1:
                    results_by_model[model_name]["r_at_1"] += 1
                if 1 <= rank <= 3:
                    results_by_model[model_name]["r_at_3"] += 1
                if rank > 0:
                    results_by_model[model_name]["mrr_sum"] += (1.0 / rank)

        # Compute summary metrics table
        summary_table = []
        
        for model_name, counts in results_by_model.items():
            num_queries = counts["evaluated"]
            if num_queries == 0:
                # No query in the dataset carries a photo, so this baseline cannot be measured
                summary_table.append({"Model": model_name, **{k: "N/A" for k in
                    ["Accuracy", "Precision", "Recall", "F1 Score", "Recall@1", "Recall@3", "MRR"]}})
                continue
            tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
            acc = (tp + tn) / max(1, (tp + fp + fn + tn))
            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = (2 * prec * rec) / max(0.0001, (prec + rec))
            r_1 = counts["r_at_1"] / max(1, num_queries)
            r_3 = counts["r_at_3"] / max(1, num_queries)
            mrr = counts["mrr_sum"] / max(1, num_queries)
            
            summary_table.append({
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "Recall@1": round(r_1, 4),
                "Recall@3": round(r_3, 4),
                "MRR": round(mrr, 4)
            })
            
        return {
            "total_eval_queries": len(lost_items),
            "comparison_matrix": summary_table
        }


if __name__ == "__main__":
    dataset = generate_dataset()
    evaluator = SystemEvaluator(dataset)
    res = evaluator.evaluate_all()
    print("--- Model Benchmark Results ---")
    import json
    print(json.dumps(res, indent=2))
