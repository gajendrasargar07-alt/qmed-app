import logging
import os
import threading

from flask import Flask, jsonify, render_template, request

from qmed import __version__, config, service
from qmed.ocr import tesseract_available

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; "
       "script-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


def create_app() -> Flask:
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024
    logging.basicConfig(level=logging.INFO)

    @app.after_request
    def headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__,
                               max_mb=config.MAX_UPLOAD_MB)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "version": __version__, "ocr": tesseract_available()})

    @app.post("/api/analyze")
    def analyze():
        cancer = request.form.get("type") or None
        upload = request.files.get("file")
        if upload is not None:
            result = service.analyze_bytes(upload.read(), cancer)
        else:
            result = service.analyze_text(request.form.get("text", ""), cancer)
        return jsonify(result)

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"status": "error", "title": "That file is too large",
                        "message": f"Choose a file under {config.MAX_UPLOAD_MB} MB."}), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"status": "error", "title": "Not found", "message": ""}), 404

    return app


app = create_app()

if __name__ == "__main__":
    from qmed.scoring import warm_up
    threading.Thread(target=warm_up, daemon=True).start()
    port = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        app.run(host="0.0.0.0", port=port)
