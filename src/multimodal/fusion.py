"""
Multimodal Feature Fusion Engine for SRM CampusFind.
Fuses Text, Image, Geo-Location, Timestamp, and Category/Attribute similarities.
Includes configurable feature weights, confidence threshold classification,
and ablation study runner.
"""

import math
from datetime import datetime
from typing import Dict, Any, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two GPS coordinates using Haversine formula."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def compute_location_similarity(coords1: list, coords2: list, max_campus_dist_m: float = 800.0) -> float:
    """Compute location similarity score using exponential spatial decay."""
    if not coords1 or not coords2 or len(coords1) < 2 or len(coords2) < 2:
        return 0.5  # Neutral default if missing
    dist_m = haversine_distance(coords1[0], coords1[1], coords2[0], coords2[1])
    # Exponential decay over campus distance (e.g., 200m yields high similarity, 800m decays to near 0)
    score = math.exp(-dist_m / 200.0)
    return max(0.0, min(1.0, float(score)))


def compute_time_similarity(ts1_str: str, ts2_str: str, half_life_hours: float = 48.0) -> float:
    """Compute temporal similarity using exponential decay over hours delta."""
    if not ts1_str or not ts2_str:
        return 0.5
    try:
        dt1 = datetime.fromisoformat(ts1_str)
        dt2 = datetime.fromisoformat(ts2_str)
        diff_hours = abs((dt1 - dt2).total_seconds()) / 3600.0
        score = math.exp(-diff_hours / half_life_hours)
        return max(0.0, min(1.0, float(score)))
    except Exception:
        return 0.5


def compute_attribute_similarity(item1: dict, item2: dict) -> float:
    """Compute categorical & attribute overlap score (Category, Brand, Color)."""
    scores = []
    
    # 1. Category match
    cat1 = str(item1.get("category", "")).lower()
    cat2 = str(item2.get("category", "")).lower()
    if cat1 and cat2:
        scores.append(1.0 if cat1 == cat2 else 0.0)
        
    # 2. Color match
    c1 = str(item1.get("color", "")).lower()
    c2 = str(item2.get("color", "")).lower()
    if c1 and c2:
        # check partial string match
        if c1 in c2 or c2 in c1:
            scores.append(1.0)
        else:
            scores.append(0.0)
            
    # 3. Brand match
    b1 = str(item1.get("brand", "")).lower()
    b2 = str(item2.get("brand", "")).lower()
    if b1 and b2 and b1 != "unknown" and b2 != "unknown":
        scores.append(1.0 if b1 == b2 else 0.0)
        
    return float(sum(scores) / len(scores)) if scores else 0.5


class MultimodalFusionEngine:
    """
    Objective 3 Multimodal Feature Fusion Engine.
    Fuses Text, Image, Location, Time, and Attribute similarity scores into a unified confidence metric.
    """
    
    DEFAULT_WEIGHTS = {
        "w_text": 0.35,
        "w_image": 0.30,
        "w_location": 0.15,
        "w_time": 0.10,
        "w_attribute": 0.10
    }
    
    HIGH_CONFIDENCE_THRESHOLD = 0.50
    POSSIBLE_MATCH_THRESHOLD = 0.40
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights if weights else dict(self.DEFAULT_WEIGHTS)
        self.normalize_weights()
        
    def normalize_weights(self):
        total = sum(self.weights.values())
        if total > 0:
            for k in self.weights:
                self.weights[k] /= total

    def qualifies_for_display(self, final_score: float) -> bool:
        return final_score >= self.POSSIBLE_MATCH_THRESHOLD

    @staticmethod
    def _weighted_average(pairs) -> float:
        """Weighted average over (weight, score) pairs, skipping missing (None) scores."""
        available = [(w, s) for w, s in pairs if s is not None]
        total_w = sum(w for w, _ in available)
        if total_w == 0:
            return 0.0
        return sum(w * s for w, s in available) / total_w

    def fuse(
        self,
        text_sim: float,
        image_sim: Optional[float],
        location_sim: float,
        time_sim: float,
        attribute_sim: float
    ) -> Dict[str, Any]:
        """
        Compute final multimodal score:
        Final Score = w1*St + w2*Si + w3*Sl + w4*Stime + w5*Sattr
        If no photo is available (image_sim is None), the visual term is dropped and
        the remaining weights are renormalized to sum to 1 (missing-modality handling).
        """
        weights_used = dict(self.weights)
        if image_sim is None:
            remaining = (1.0 - weights_used["w_image"]) or 1.0
            weights_used = {k: (0.0 if k == "w_image" else v / remaining) for k, v in weights_used.items()}

        final_score = self._weighted_average([
            (self.weights["w_text"], text_sim),
            (self.weights["w_image"], image_sim),
            (self.weights["w_location"], location_sim),
            (self.weights["w_time"], time_sim),
            (self.weights["w_attribute"], attribute_sim)
        ])
        
        # Determine confidence classification label
        if final_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            classification = "HIGH CONFIDENCE MATCH"
        elif final_score >= self.POSSIBLE_MATCH_THRESHOLD:
            classification = "POSSIBLE MATCH"
        else:
            classification = "LOW CONFIDENCE / NO MATCH"
            
        return {
            "final_score": round(float(final_score), 4),
            "classification": classification,
            "subscores": {
                "text_similarity": round(float(text_sim), 4),
                "image_similarity": None if image_sim is None else round(float(image_sim), 4),
                "location_similarity": round(float(location_sim), 4),
                "time_similarity": round(float(time_sim), 4),
                "attribute_similarity": round(float(attribute_sim), 4)
            },
            "weights_used": {k: round(v, 4) for k, v in weights_used.items()}
        }

    def run_ablation(
        self,
        text_sim: float,
        image_sim: Optional[float],
        location_sim: float,
        time_sim: float,
        attribute_sim: float
    ) -> Dict[str, Optional[float]]:
        """Run Ablation Study showing performance across feature subsets (image terms skipped if no photo)."""
        avg = self._weighted_average
        return {
            "Text Only": round(text_sim, 4),
            "Image Only": None if image_sim is None else round(image_sim, 4),
            "Text + Image": round(avg([(0.55, text_sim), (0.45, image_sim)]), 4),
            "Text + Image + Location": round(avg([(0.45, text_sim), (0.35, image_sim), (0.20, location_sim)]), 4),
            "Text + Image + Location + Time": round(avg([(0.40, text_sim), (0.30, image_sim), (0.15, location_sim), (0.15, time_sim)]), 4),
            "Full Multimodal": self.fuse(text_sim, image_sim, location_sim, time_sim, attribute_sim)["final_score"]
        }


if __name__ == "__main__":
    fusion = MultimodalFusionEngine()
    result = fusion.fuse(
        text_sim=0.94, 
        image_sim=0.89, 
        location_sim=0.93, 
        time_sim=0.86, 
        attribute_sim=1.0
    )
    print("--- Multimodal Fusion Engine Test ---")
    print(f"Final Score:    {result['final_score']}")
    print(f"Classification: {result['classification']}")
    print("Subscores:", result["subscores"])
    print("Ablation Study:", fusion.run_ablation(0.94, 0.89, 0.93, 0.86, 1.0))
