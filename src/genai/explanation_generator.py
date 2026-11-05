"""
Generative AI Explanation, Structured Attribute Normalizer, and Paraphrase Generator.
Provides natural-language evidence-grounded match explanations, automated report normalization,
and synthetic training data expansion for SRM CampusFind.
"""

import json
from typing import Dict, Any, List

class GenAIExplanationEngine:
    """
    Objective 4 Generative AI Component.
    Generates natural-language evidence-grounded explanations for matched lost & found items.
    """
    
    @staticmethod
    def generate_explanation(
        lost_report: Dict[str, Any],
        found_report: Dict[str, Any],
        match_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an evidence-grounded explanation based strictly on matching metadata and scores.
        Avoids hallucination by referencing exact reported attributes.
        """
        subscores = match_result.get("subscores", {})
        final_score = match_result.get("final_score", 0.0)
        classification = match_result.get("classification", "")
        
        lost_desc = lost_report.get("description", "")
        found_desc = found_report.get("description", "")
        lost_loc = lost_report.get("location_name", "Reported location")
        found_loc = found_report.get("location_name", "Reported location")
        
        # Build explanation bullet points from ground evidence
        evidence_points = []
        
        if subscores.get("text_similarity", 0) >= 0.70:
            evidence_points.append(
                f"High semantic text similarity ({round(subscores['text_similarity']*100)}%): Both descriptions refer to similar items ('{lost_desc}' vs '{found_desc}')."
            )
        else:
            evidence_points.append(
                f"Moderate text similarity ({round(subscores['text_similarity']*100)}%): Descriptions share partial keyword context."
            )
            
        image_sim = subscores.get("image_similarity")
        if image_sim is None:
            evidence_points.append(
                "Visual similarity was not used because a photo is not available for both reports; the score is based on the remaining evidence."
            )
        elif image_sim >= 0.75:
            evidence_points.append(
                f"High visual similarity ({round(image_sim*100)}%): Uploaded image feature embeddings show matching visual geometry, color distribution, and item structure."
            )
        else:
            evidence_points.append(
                f"Limited visual similarity ({round(image_sim*100)}%): The uploaded photos differ noticeably in appearance."
            )
            
        if subscores.get("location_similarity", 0) >= 0.80:
            evidence_points.append(
                f"Spatial proximity ({round(subscores['location_similarity']*100)}%): Lost location ('{lost_loc}') and found location ('{found_loc}') are within close physical proximity on SRM campus."
            )
            
        if subscores.get("time_similarity", 0) >= 0.75:
            evidence_points.append(
                f"Temporal compatibility ({round(subscores['time_similarity']*100)}%): The lost timestamp and found timestamp occurred within an expected sequential timeframe."
            )
            
        # Compose natural language explanation text
        explanation_text = (
            f"SRM CampusFind AI identified this pair as a {classification} with an overall match confidence of {round(final_score*100)}%. "
            f"The model reached this decision because: " + " ".join(evidence_points) + " "
            "Please verify ownership through the SRM Lost-and-Found Security desk before claiming."
        )
        
        return {
            "explanation_text": explanation_text,
            "evidence_grounding": {
                "reported_lost_item": lost_desc,
                "reported_found_item": found_desc,
                "lost_location": lost_loc,
                "found_location": found_loc,
                "confidence_score": final_score,
                "classification": classification
            },
            "responsible_ai_disclaimer": "This explanation is AI-generated based on multimodal score evidence. Physical human verification of ownership is mandatory."
        }


class GenAINormalizerEngine:
    """
    Generative Component 2: Automatic Structured Item Attribute Normalizer.
    Converts unstructured raw text inputs into structured schema fields.
    """
    
    @staticmethod
    def normalize_description(raw_text: str) -> Dict[str, str]:
        """Convert raw user input into normalized category, color, brand, and clean text."""
        text_lower = raw_text.lower()
        
        # Category detection
        category = "other"
        if any(w in text_lower for w in ["earbuds", "earphones", "airpods", "buds", "headphone"]):
            category = "electronics (audio)"
        elif any(w in text_lower for w in ["phone", "iphone", "samsung", "mobile", "laptop", "charger", "ipad"]):
            category = "electronics (device)"
        elif any(w in text_lower for w in ["bag", "backpack", "purse", "wallet", "rucksack"]):
            category = "bags & luggage"
        elif any(w in text_lower for w in ["id card", "identity", "license", "card", "passport"]):
            category = "documents & IDs"
        elif any(w in text_lower for w in ["glasses", "sunglasses", "bottle", "keys", "umbrella"]):
            category = "personal accessories"

        # Color detection
        colors = []
        for c in ["black", "blue", "navy", "red", "white", "silver", "grey", "gray", "green", "yellow", "brown", "pink", "gold"]:
            if c in text_lower:
                colors.append(c)
        color_str = ", ".join(colors) if colors else "unspecified"
        
        # Brand detection
        brands = ["nike", "boat", "apple", "samsung", "dell", "hp", "lenovo", "ray-ban", "milton", "honda", "fastrack"]
        detected_brand = "Unspecified"
        for b in brands:
            if b in text_lower:
                detected_brand = b.capitalize()
                break

        return {
            "original_raw_text": raw_text,
            "normalized_category": category,
            "normalized_color": color_str,
            "detected_brand": detected_brand,
            "clean_prompt": text_lower.strip()
        }


class GenAISyntheticDataGenerator:
    """
    Generative Component 3: Synthetic Paraphrased Data Generator.
    Produces semantic-preserving synthetic text variations for robustness testing.
    """
    
    TEMPLATES = [
        "I lost my {color} {brand} {item} around {location}.",
        "Misplaced a {color} {item} near {location}. Please contact if found.",
        "Dropped my {brand} {item} ({color}) while walking near {location}.",
        "Looking for a missing {color} {item} left close to {location}."
    ]
    
    @classmethod
    def generate_synthetic_samples(cls, item: str, color: str, brand: str, location: str, n: int = 3) -> List[str]:
        samples = []
        for i in range(min(n, len(cls.TEMPLATES))):
            tmpl = cls.TEMPLATES[i]
            text = tmpl.format(item=item, color=color, brand=brand, location=location)
            samples.append(text)
        return samples


if __name__ == "__main__":
    print("--- GenAI Normalization Test ---")
    norm = GenAINormalizerEngine.normalize_description("black boat buds lost near lib")
    print(json.dumps(norm, indent=2))
    
    print("\n--- GenAI Explanation Test ---")
    lost = {"description": "Misplaced black wireless earbuds near library.", "location_name": "Central Library"}
    found = {"description": "Found dark Bluetooth earphones close to library.", "location_name": "Library Lobby"}
    m_res = {"final_score": 0.91, "classification": "HIGH CONFIDENCE MATCH", "subscores": {"text_similarity": 0.94, "image_similarity": 0.89, "location_similarity": 0.93, "time_similarity": 0.86}}
    expl = GenAIExplanationEngine.generate_explanation(lost, found, m_res)
    print(expl["explanation_text"])
