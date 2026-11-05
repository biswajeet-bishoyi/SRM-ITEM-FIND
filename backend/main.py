"""
FastAPI Backend Server for SRM CampusFind.
Provides REST APIs for lost/found item submission, multimodal similarity search,
GenAI explanations, vector retrieval, and automated notifications.
"""

import os
import sys
import json
import traceback
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    err = traceback.format_exc()
    print(f"[GLOBAL UNHANDLED EXCEPTION]: {err}")
    return JSONResponse(
        status_code=500,
        content={"error": "Server Error", "detail": str(exc), "traceback": err}
    )

if os.environ.get("VERCEL"):
    UPLOADS_DIR = os.path.join("/tmp", "uploads")
else:
    UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

try:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
except OSError:
    UPLOADS_DIR = os.path.join("/tmp", "uploads")
    os.makedirs(UPLOADS_DIR, exist_ok=True)

USER_REPORTS_FILE = os.path.join(UPLOADS_DIR, "user_submitted_records.json")

def persist_user_record(rec: Dict[str, Any]):
    try:
        existing = []
        if os.path.exists(USER_REPORTS_FILE):
            with open(USER_REPORTS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing = [r for r in existing if r.get("id") != rec.get("id")]
        existing.append(rec)
        with open(USER_REPORTS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f)
    except Exception as e:
        print(f"[persist_user_record] Note: {e}")

def load_user_records() -> List[Dict[str, Any]]:
    if os.path.exists(USER_REPORTS_FILE):
        try:
            with open(USER_REPORTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

startup_error = None
dataset = []
text_encoder = None
image_encoder = None
fusion_engine = None
notification_manager = None
vector_db = None

is_initialized = False

def init_app():
    global is_initialized, dataset, text_encoder, image_encoder, fusion_engine, notification_manager, vector_db, startup_error
    if is_initialized:
        return
    try:
        from src.preprocessing.text_cleaner import clean_text, extract_structured_attributes
        from src.preprocessing.dataset_generator import generate_dataset
        from src.text_matching.encoder import TransformerTextEncoder, compute_cosine_similarity
        from src.image_matching.encoder import ImageEncoder
        from src.multimodal.fusion import MultimodalFusionEngine, compute_location_similarity, compute_time_similarity, compute_attribute_similarity
        from src.retrieval.vector_db import VectorDatabase
        from src.genai.explanation_generator import GenAIExplanationEngine, GenAINormalizerEngine, GenAISyntheticDataGenerator
        from src.notification.notifier import NotificationManager

        dataset = generate_dataset(PROJECT_ROOT)
        text_encoder = TransformerTextEncoder()
        image_encoder = ImageEncoder()
        fusion_engine = MultimodalFusionEngine()
        notification_manager = NotificationManager(confidence_threshold=0.60)
        vector_db = VectorDatabase()

        corpus = [r["description"] for r in dataset]
        text_encoder.fit_fallback(corpus)

        cache_json = os.path.join(PROJECT_ROOT, "data", "processed", "precomputed_embeddings.json")
        cache_npz = os.path.join(PROJECT_ROOT, "data", "processed", "precomputed_embeddings.npz")
        loaded_from_cache = False
        
        # Prefer JSON cache to avoid numpy version mismatch issues on cloud servers
        if os.path.exists(cache_json):
            try:
                import json
                with open(cache_json, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                t_embs = cached.get("text_embs", [])
                i_embs = cached.get("image_embs", [])
                if len(t_embs) == len(dataset):
                    import numpy as np
                    for idx, item in enumerate(dataset):
                        t_emb = np.array(t_embs[idx], dtype=np.float32)
                        raw_i = np.array(i_embs[idx], dtype=np.float32) if idx < len(i_embs) else None
                        i_emb = raw_i if (raw_i is not None and np.any(raw_i)) else None
                        vector_db.add_record(item, t_emb, i_emb)
                    loaded_from_cache = True
            except Exception as ce:
                print(f"[init_app] JSON cache load error: {ce}")

        if not loaded_from_cache and os.path.exists(cache_npz):
            try:
                import numpy as np
                cached = np.load(cache_npz)
                t_embs = cached["text_embs"]
                i_embs = cached["image_embs"]
                if len(t_embs) == len(dataset):
                    for idx, item in enumerate(dataset):
                        t_emb = t_embs[idx]
                        i_emb = i_embs[idx] if np.any(i_embs[idx]) else None
                        vector_db.add_record(item, t_emb, i_emb)
                    loaded_from_cache = True
            except Exception as ce:
                print(f"[init_app] NPZ cache load error: {ce}")

        if not loaded_from_cache:
            for item in dataset:
                t_emb = text_encoder.encode(item["description"])
                i_emb = image_encoder.encode(item["image_filename"])
                vector_db.add_record(item, t_emb, i_emb)

        # Ingest any previously submitted user records
        for urec in load_user_records():
            if not any(r.get("id") == urec.get("id") for r in vector_db.records):
                try:
                    u_img = ImageEncoder.decode_data_url(urec.get("image_url") or urec.get("image_data"))
                    desc = urec.get("description") or f"{urec.get('brand', '')} {urec.get('category', 'item')}"
                    u_t_emb = text_encoder.encode(desc)
                    u_i_emb = image_encoder.encode(u_img) if u_img else None
                    vector_db.add_record(urec, u_t_emb, u_i_emb)
                except Exception as ue:
                    print(f"[init_app] User record load note: {ue}")

        is_initialized = True
    except Exception as e:
        startup_error = traceback.format_exc()
        print(f"[FATAL STARTUP ERROR]: {startup_error}")

@app.on_event("startup")
def on_startup():
    init_app()


class ReportItemRequest(BaseModel):
    status: str  # "LOST" or "FOUND"
    description: Optional[str] = ""
    category: Optional[str] = "other"
    brand: Optional[str] = "unknown"
    color: Optional[str] = "unspecified"
    location_name: str
    coordinates: Optional[List[float]] = [12.8231, 80.0442]
    timestamp: Optional[str] = None
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
    user_records: Optional[List[Dict[str, Any]]] = None


@app.get("/api/health")
@app.get("/health")
def health_check():
    init_app()
    if startup_error:
        return {
            "status": "initialization_error",
            "system": "SRM CampusFind AI Backend",
            "error": startup_error,
            "environment": "vercel" if os.environ.get("VERCEL") else "local"
        }
    return {
        "status": "online",
        "system": "SRM CampusFind AI Backend",
        "dataset_size": len(vector_db.records) if vector_db else 0,
        "environment": "vercel" if os.environ.get("VERCEL") else "local"
    }


@app.get("/api/dataset")
@app.get("/dataset")
def get_dataset():
    init_app()
    records = vector_db.records if vector_db else []
    return {"total_records": len(records), "records": records}


def build_report_record(report: ReportItemRequest, id_prefix: str) -> Dict[str, Any]:
    """Create a record from a submitted report, saving its photo (if any) and indexing its embeddings."""
    init_app()
    new_id = f"{id_prefix}_{len(vector_db.records) + 101}" if vector_db else f"{id_prefix}_101"
    
    # Support both Pydantic v1 (.dict) and v2 (.model_dump)
    if hasattr(report, "model_dump"):
        record = report.model_dump(exclude={"image_data"})
    else:
        record = report.dict(exclude={"image_data"})
        
    record["id"] = new_id
    record["data_source"] = "USER_SUBMITTED"

    # Ensure description is never empty so text embedding never fails
    desc = (record.get("description") or "").strip()
    if not desc:
        parts = [record.get("color"), record.get("brand"), record.get("category"), f"near {record.get('location_name', 'campus')}"]
        desc = " ".join([p for p in parts if p and p not in ("unknown", "unspecified", "other")])
        if not desc:
            desc = f"{id_prefix.lower()} item reported at {record.get('location_name', 'SRM campus')}"
        record["description"] = desc

    if not record.get("timestamp"):
        from datetime import datetime
        record["timestamp"] = datetime.now().isoformat()

    img = None
    if report.image_data:
        try:
            img = ImageEncoder.decode_data_url(report.image_data)
        except Exception as de:
            print(f"[decode_data_url warning]: {de}")

    if img is not None:
        try:
            import io, base64
            img.thumbnail((450, 450))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/jpeg;base64,{b64_str}"
            record["image_url"] = data_url
            record["image_data"] = data_url
        except Exception as te:
            print(f"[thumbnail warning]: {te}")
            record["image_url"] = report.image_data
            record["image_data"] = report.image_data

        try:
            filepath = os.path.join(UPLOADS_DIR, f"{new_id}.jpg")
            img.save(filepath, "JPEG", quality=85)
            record["image_filename"] = filepath
        except Exception:
            record["image_filename"] = record.get("image_url")

    try:
        if text_encoder and vector_db:
            t_emb = text_encoder.encode(record["description"])
            i_emb = image_encoder.encode(img) if (image_encoder and img) else None
            vector_db.add_record(record, t_emb, i_emb)
    except Exception as ee:
        print(f"[vector_db indexing warning]: {ee}")

    persist_user_record(record)
    return record


def check_for_cross_matches(new_record: Dict[str, Any], target_status: str, min_confidence: float = 0.60) -> List[Dict[str, Any]]:
    """Automatically find existing counter-reports matching this new report with >= min_confidence and trigger alerts."""
    init_app()
    alerts = []
    if not vector_db:
        return alerts
    
    try:
        from src.multimodal.fusion import compute_location_similarity, compute_time_similarity, compute_attribute_similarity
        from src.genai.explanation_generator import GenAIExplanationEngine
        from src.image_matching.encoder import ImageEncoder
        
        t_emb = text_encoder.encode(new_record["description"])
        img = ImageEncoder.decode_data_url(new_record.get("image_data")) if new_record.get("image_data") else None
        i_emb = image_encoder.encode(img) if img else None
        
        candidates = vector_db.search(t_emb[0], i_emb, target_status=target_status, top_k=5)
        for cand, t_sim, i_sim in candidates:
            if cand.get("id") == new_record.get("id"):
                continue
            loc_sim = compute_location_similarity(new_record.get("coordinates", [12.8231, 80.0442]), cand.get("coordinates", [12.8231, 80.0442]))
            time_sim = compute_time_similarity(new_record.get("timestamp", ""), cand.get("timestamp", ""))
            attr_sim = compute_attribute_similarity(new_record, cand)
            
            fused = fusion_engine.fuse(t_sim, i_sim, loc_sim, time_sim, attr_sim)
            score = fused["final_score"]
            
            if score >= min_confidence:
                explanation = GenAIExplanationEngine.generate_explanation(new_record, cand, fused)
                notif = notification_manager.process_match_result(new_record, cand, fused, explanation)
                alerts.append({
                    "match_score": round(float(score), 4),
                    "confidence_percent": int(round(float(score) * 100)),
                    "matched_item": cand,
                    "explanation": explanation.get("explanation_text", ""),
                    "notification_id": notif.get("notification_id")
                })
    except Exception as ce:
        print(f"[check_for_cross_matches warning]: {ce}")
    return alerts


@app.post("/api/report/lost")
@app.post("/report/lost")
def report_lost_item(report: ReportItemRequest):
    init_app()
    try:
        from src.genai.explanation_generator import GenAINormalizerEngine
        if report.category == "other" and report.description:
            try:
                report.category = GenAINormalizerEngine.normalize_description(report.description).get("normalized_category", report.category)
            except Exception:
                pass
        record = build_report_record(report, "LOST")
        
        # Automated notification triggered when match >= 60% (failsafe: will not fail report if cross-match check has an issue)
        high_conf_matches = []
        try:
            high_conf_matches = check_for_cross_matches(record, target_status="FOUND", min_confidence=0.60)
        except Exception as cme:
            print(f"[report_lost_item match warning]: {cme}")
        
        return {
            "message": "Lost report created successfully",
            "record": record,
            "high_confidence_matches": high_conf_matches,
            "alert_triggered": len(high_conf_matches) > 0
        }
    except Exception as e:
        err = traceback.format_exc()
        print(f"[report_lost_item error]: {err}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/report/found")
@app.post("/report/found")
def report_found_item(report: ReportItemRequest):
    init_app()
    try:
        record = build_report_record(report, "FOUND")
        
        # Automated notification check for owner when match >= 60%
        high_conf_matches = []
        try:
            high_conf_matches = check_for_cross_matches(record, target_status="LOST", min_confidence=0.60)
        except Exception as cme:
            print(f"[report_found_item match warning]: {cme}")
        
        return {
            "message": "Found report created successfully",
            "record": record,
            "high_confidence_matches": high_conf_matches,
            "alert_triggered": len(high_conf_matches) > 0
        }
    except Exception as e:
        err = traceback.format_exc()
        print(f"[report_found_item error]: {err}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/matches")
@app.post("/search/matches")
def search_matches(req: SearchMatchRequest):
    init_app()
    if startup_error:
        raise HTTPException(status_code=500, detail=f"AI backend initialization failed: {startup_error}")

    # Ensure any client-side user records are indexed in vector_db (supports serverless lambda resets)
    if req.user_records:
        for urec in req.user_records:
            if not any(r.get("id") == urec.get("id") for r in vector_db.records):
                u_img = ImageEncoder.decode_data_url(urec.get("image_url") or urec.get("image_data"))
                u_t_emb = text_encoder.encode(urec["description"])
                u_i_emb = image_encoder.encode(u_img) if u_img else None
                vector_db.add_record(urec, u_t_emb, u_i_emb)

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
@app.get("/notifications")
def get_notifications():
    return {"notifications": notification_manager.get_all_notifications() if notification_manager else []}


@app.post("/api/synthetic/generate")
@app.post("/synthetic/generate")
def generate_synthetic_samples(item: str, color: str, brand: str, location: str):
    from src.genai.explanation_generator import GenAISyntheticDataGenerator
    variations = GenAISyntheticDataGenerator.generate_synthetic_samples(item, color, brand, location)
    return {"original": f"{color} {brand} {item} lost near {location}", "synthetic_paraphrases": variations}


# Serve Frontend Web App and uploaded item photos (optional for local run)
try:
    from fastapi.staticfiles import StaticFiles
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    images_dir = os.path.join(PROJECT_ROOT, "data", "images")
    if os.path.exists(images_dir):
        app.mount("/data/images", StaticFiles(directory=images_dir), name="images")
        app.mount("/images", StaticFiles(directory=images_dir), name="images_alt")

    if os.path.exists(UPLOADS_DIR):
        app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
except Exception as sfe:
    print(f"[StaticFiles mounting note]: {sfe}")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    if startup_error:
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>SRM CampusFind — Initialization Error</title></head>
        <body style="font-family:system-ui,-apple-system,sans-serif;background:#090d16;color:#f87171;padding:40px;line-height:1.5;">
            <h1 style="color:#ef4444;">SRM CampusFind — Backend Initialization Error</h1>
            <p style="color:#94a3b8;">An error occurred during component startup:</p>
            <pre style="background:#131d2e;color:#fca5a5;padding:20px;border-radius:8px;overflow-x:auto;border:1px solid #334155;">{startup_error}</pre>
        </body>
        </html>
        """
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>SRM CampusFind API Running. Frontend index.html not found.</h1>"
