"""What the web layer calls: bytes or text in, display-ready result out."""
import logging

from . import config, pipeline
from .ocr import extract
from .presenter import present

log = logging.getLogger("qmed")


def _error(title: str, message: str) -> dict:
    return {"status": "error", "title": title, "message": message}


def _run(source, kind: str, cancer: str = None) -> dict:
    if cancer is not None and cancer not in config.CANCERS:
        cancer = None
    try:
        return present(pipeline.analyze(source, kind, cancer))
    except Exception:
        log.exception("analysis failed")
        return _error("Something went wrong",
                      "We couldn't analyse this report. Try again, or upload a different file.")


def analyze_bytes(data: bytes, cancer: str = None) -> dict:
    if not data:
        return _error("That file is empty", "Choose a report and try again.")
    return _run(data, "auto", cancer)


def analyze_text(text: str, cancer: str = None) -> dict:
    if not text or not text.strip():
        return _error("Nothing to analyse", "Paste the text of your report and try again.")
    return _run(text, "text", cancer)
