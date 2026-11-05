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

    @app.get("/{full_path:path}")
    def error_handler(full_path: str):
        return HTMLResponse(
            content=f"""
            <html>
            <body style="font-family:sans-serif;background:#0f172a;color:#f87171;padding:40px;">
                <h2>Vercel Startup Exception in backend.main</h2>
                <pre style="background:#1e293b;padding:20px;border-radius:8px;overflow:auto;color:#cbd5e1;">{err}</pre>
            </body>
            </html>
            """,
            status_code=500
        )

