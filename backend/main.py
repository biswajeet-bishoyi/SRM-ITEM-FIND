"""
FastAPI Backend Server for SRM CampusFind.
Provides REST APIs for lost/found item submission, multimodal similarity search,
GenAI explanations, vector retrieval, and automated notifications.
"""

import os
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.preprocessing.text_cleaner import clean_text, extract_structured_attributes
from src.preprocessing.dataset_generator import generate_dataset
from src.text_matching.encoder import TransformerTextEncoder, compute_cosine_similarity
from src.image_matching.encoder import ImageEncoder
from src.multimodal.fusion import MultimodalFusionEngine, compute_location_similarity, compute_time_similarity, compute_attribute_similarity
from src.retrieval.vector_db import VectorDatabase
from src.genai.explanation_generator import GenAIExplanationEngine, GenAINormalizerEngine, GenAISyntheticDataGenerator
from src.notification.notifier import NotificationManager


app = FastAPI(
    title="SRM CampusFind API",
    description="Intelligent Multimodal Lost-and-Found System using GenAI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize Core AI Components
dataset = generate_dataset()
text_encoder = TransformerTextEncoder()
image_encoder = ImageEncoder()
fusion_engine = MultimodalFusionEngine()
notification_manager = NotificationManager(confidence_threshold=0.50)
vector_db = VectorDatabase()

# Populate Vector DB
corpus = [r["description"] for r in dataset]
text_encoder.fit_fallback(corpus)

for item in dataset:
    t_emb = text_encoder.encode(item["description"])
    i_emb = image_encoder.encode(item["image_filename"])
    vector_db.add_record(item, t_emb, i_emb)


class ReportItemRequest(BaseModel):
    status: str  # "LOST" or "FOUND"
    description: str
    category: Optional[str] = "other"
    brand: Optional[str] = "unknown"
    color: Optional[str] = "unspecified"
    location_name: str
    coordinates: Optional[List[float]] = [12.8231, 80.0442]
    timestamp: str
    image_filename: Optional[str] = None
    image_data: Optional[str] = None  # Optional photo as a base64 data URL


class SearchMatchRequest(BaseModel):
    query_description: str
    target_status: str = "FOUND"
    location_name: Optional[str] = "Reported location"
    category: Optional[str] = None
    color: Optional[str] = None
    brand: Optional[str] = None
    coordinates: Optional[List[float]] = [12.8231, 80.0442]
    timestamp: Optional[str] = "2026-10-05T12:00:00"
    image_data: Optional[str] = None  # Optional photo as a base64 data URL
    weights: Optional[Dict[str, float]] = None


@app.get("/api/health")
def health_check():
    return {"status": "online", "system": "SRM CampusFind AI Backend", "dataset_size": len(vector_db.records)}


@app.get("/api/dataset")
def get_dataset():
    return {"total_records": len(vector_db.records), "records": vector_db.records}


def build_report_record(report: ReportItemRequest, id_prefix: str) -> Dict[str, Any]:
    """Create a record from a submitted report, saving its photo (if any) and indexing its embeddings."""
    new_id = f"{id_prefix}_{len(vector_db.records) + 101}"
    record = report.model_dump(exclude={"image_data"})
    record["id"] = new_id
    record["data_source"] = "USER_SUBMITTED"

    img = ImageEncoder.decode_data_url(report.image_data)
    if report.image_data and img is None:
        raise HTTPException(status_code=400, detail="Uploaded photo could not be read as an image.")
    if img is not None:
        img.thumbnail((800, 800))
        img.save(os.path.join(UPLOADS_DIR, f"{new_id}.jpg"), "JPEG", quality=85)
        record["image_filename"] = os.path.join(UPLOADS_DIR, f"{new_id}.jpg")
        record["image_url"] = f"/uploads/{new_id}.jpg"

    t_emb = text_encoder.encode(record["description"])
    i_emb = image_encoder.encode(img)
    vector_db.add_record(record, t_emb, i_emb)
    return record


@app.post("/api/report/lost")
def report_lost_item(report: ReportItemRequest):
    # Run GenAI normalization
    if report.category == "other":
        report.category = GenAINormalizerEngine.normalize_description(report.description)["normalized_category"]
    record = build_report_record(report, "LOST")
    return {"message": "Lost report created successfully", "record": record}


@app.post("/api/report/found")
def report_found_item(report: ReportItemRequest):
    record = build_report_record(report, "FOUND")
    return {"message": "Found report created successfully", "record": record}


@app.post("/api/search/matches")
def search_matches(req: SearchMatchRequest):
    """
    Multimodal Match Search API:
    1. Encodes query text & image.
    2. Retrieves top-K nearest candidates from Vector DB.
     3. Computes multimodal similarity scores and keeps candidates with at least
         40% fused confidence.
     4. Generates GenAI evidence explanations for qualifying candidates.
     5. Triggers notification if score >= threshold.
    """
    q_t_emb = text_encoder.encode(req.query_description)
    query_img = ImageEncoder.decode_data_url(req.image_data)
    if req.image_data and query_img is None:
        raise HTTPException(status_code=400, detail="Uploaded photo could not be read as an image.")
    q_i_emb = image_encoder.encode(query_img)  # None when no photo was uploaded
    
    # Configure custom fusion weights if provided
    custom_fusion = MultimodalFusionEngine(req.weights) if req.weights else fusion_engine
    
    # 1. Vector Retrieval
    candidates = vector_db.search(q_t_emb[0], q_i_emb, target_status=req.target_status, top_k=5)
    
    match_results = []
    
    query_record = {
        "description": req.query_description,
        "category": req.category,
        "color": req.color,
        "brand": req.brand,
        "location_name": req.location_name,
        "coordinates": req.coordinates,
        "timestamp": req.timestamp,
        "has_photo": query_img is not None
    }
    
    for rec, t_sim, i_sim in candidates:
        loc_sim = compute_location_similarity(req.coordinates, rec.get("coordinates", [12.8231, 80.0442]))
        time_sim = compute_time_similarity(req.timestamp, rec.get("timestamp", "2026-10-05T12:00:00"))
        attr_sim = compute_attribute_similarity(query_record, rec)
        
        # 2. Multimodal Fusion
        fused = custom_fusion.fuse(t_sim, i_sim, loc_sim, time_sim, attr_sim)

        if not custom_fusion.qualifies_for_display(fused["final_score"]):
            continue
        
        # 3. GenAI Explanation
        explanation = GenAIExplanationEngine.generate_explanation(query_record, rec, fused)
        
        # 4. Notification Check
        notif = notification_manager.process_match_result(query_record, rec, fused, explanation)
        
        # 5. Ablation Breakdown
        ablation = custom_fusion.run_ablation(t_sim, i_sim, loc_sim, time_sim, attr_sim)
        
        match_results.append({
            "candidate_record": rec,
            "multimodal_score": fused["final_score"],
            "classification": fused["classification"],
            "subscores": fused["subscores"],
            "ablation_study": ablation,
            "genai_explanation": explanation["explanation_text"],
            "evidence_grounding": explanation["evidence_grounding"],
            "notification_alert": notif["triggered"]
        })
        
    # Sort final output by multimodal score
    match_results.sort(key=lambda x: x["multimodal_score"], reverse=True)
    return {
        "query": query_record,
        "total_matches_found": len(match_results),
        "matches": match_results
    }


@app.get("/api/notifications")
def get_notifications():
    return {"notifications": notification_manager.get_all_notifications()}


@app.post("/api/synthetic/generate")
def generate_synthetic_samples(item: str, color: str, brand: str, location: str):
    variations = GenAISyntheticDataGenerator.generate_synthetic_samples(item, color, brand, location)
    return {"original": f"{color} {brand} {item} lost near {location}", "synthetic_paraphrases": variations}


# Serve Frontend Web App and uploaded item photos
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SRM CampusFind API Running. Frontend index.html not found.</h1>"
