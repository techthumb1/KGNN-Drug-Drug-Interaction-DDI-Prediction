import os
import sys
from pathlib import Path

from flask import Flask

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR / "src"))

if os.environ.get("FLASK_ENV") != "production":
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

app = Flask(
    __name__,
    template_folder=BASE_DIR / "backend" / "app" / "templates",
    static_folder=BASE_DIR / "backend" / "app" / "static",
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")

try:
    from src.utils.data_loader import ensure_required_files

    ensure_required_files()

    from backend.app.routes import init_routes

    init_routes(app)

    from src.utils import model as runtime
except Exception as error:
    app.logger.error("Failed to register routes: %s", error)
    raise


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "graph": "ready",
        "predictions": (
            "ready"
            if runtime.prediction_error is None
            else "unavailable"
        ),
        "explanations": "unavailable",
    }, 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        debug=False,
    )