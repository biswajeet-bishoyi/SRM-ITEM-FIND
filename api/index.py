import os
import sys
import json
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

boot_error = None
fastapi_app = None

try:
    from backend.main import app as main_app
    fastapi_app = main_app
except Exception as e:
    boot_error = traceback.format_exc()
    print(f"[FATAL VERCEL BOOT ERROR]: {boot_error}", file=sys.stderr)


async def app(scope, receive, send):
    """
    Universal ASGI entrypoint for Vercel.
    Executes FastAPI app or returns structured JSON diagnostic if boot failed.
    """
    if scope["type"] == "http":
        if fastapi_app is not None:
            try:
                await fastapi_app(scope, receive, send)
                return
            except Exception as exc:
                err_text = traceback.format_exc()
                print(f"[ASGI RUNTIME ERROR]: {err_text}", file=sys.stderr)
                body = json.dumps({
                    "error": "FastAPI ASGI Runtime Error",
                    "details": str(exc),
                    "traceback": err_text
                }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [
                        [b"content-type", b"application/json"],
                        [b"content-length", str(len(body)).encode("utf-8")],
                        [b"access-control-allow-origin", b"*"],
                    ],
                })
                await send({
                    "type": "http.response.body",
                    "body": body,
                })
                return
        else:
            path = scope.get("path", "")
            status_code = 200 if (path.endswith("/health") or path == "/api") else 500
            body = json.dumps({
                "status": "boot_diagnostic",
                "error": "Backend Startup Incomplete",
                "message": "Components could not be imported during serverless cold start.",
                "python_version": sys.version,
                "boot_error": boot_error
            }, indent=2).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode("utf-8")],
                    [b"access-control-allow-origin", b"*"],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return
    elif scope["type"] == "lifespan":
        if fastapi_app is not None:
            try:
                await fastapi_app(scope, receive, send)
                return
            except Exception:
                pass
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
