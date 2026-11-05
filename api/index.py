import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from backend.main import app
except Exception as e:
    err = traceback.format_exc()
    print(f"[FATAL VERCEL BOOT ERROR]: {err}", file=sys.stderr)
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="SRM CampusFind Diagnostic Fallback")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    def error_handler(full_path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Backend Startup Failed",
                "message": "An error occurred while loading SRM CampusFind components.",
                "details": err
            }
        )

