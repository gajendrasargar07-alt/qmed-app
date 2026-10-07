"""Turns a pipeline report into the plain-language result the app displays."""
from . import config
from .labels import CANCER_NAMES, MODEL_NAMES, format_value, label_for

BAND_KEYS = {"HIGH RISK": "high", "MODERATE RISK": "moderate", "LOW RISK": "low"}
ROMAN = {0: "0", 1: "I", 2: "II", 3: "III", 4: "IV"}


def _level(x: float) -> str:
    if x >= config.HIGH_THRESHOLD:
        return "high"
    if x >= config.MODERATE_THRESHOLD:
        return "moderate"
    return "low"


def _types():
    return [{"id": c, "label": CANCER_NAMES[c]["label"]} for c in config.CANCERS]


def _verdict_copy(band: str, label: str, condition: str) -> dict:
    if band == "high":
        return {
            "headline": "High-risk pattern found",
            "summary": (f"The values in this {label.lower()} report match patterns our models "
                        f"associate with {condition}. This is an estimate, not a diagnosis."),
            "action": "Book a follow-up with your doctor soon.",
            "steps": ["Take this report and your original results to your appointment.",
                      "Ask which tests can confirm or rule out a diagnosis."],
        }
    if band == "moderate":
        return {
            "headline": "Some values need a closer look",
            "summary": (f"The signals in this {label.lower()} report are mixed. Some values sit in a "
                        f"range linked with {condition}, and others do not."),
            "action": "Talk to your doctor about repeat tests or extra screening.",
            "steps": ["Share this report with your doctor and ask what to repeat or add.",
                      "Note any symptoms you have so you can mention them."],
        }
    return {
        "headline": "No strong risk pattern found",
        "summary": (f"The values in this {label.lower()} report do not match the patterns our models "
                    f"associate with {condition}. A low score is not a guarantee of good health."),
        "action": "Keep up with your routine check-ups.",
        "steps": ["Tell your doctor about any new or changing symptoms, whatever this score says."],
    }


def _values(report: dict, cancer: str) -> list:
    f = report["fields"]
    key = set(f["key_fields"])
    present = f["extracted"]
    ordered = ([n for n in f["required"] if n in key and n in present] +
               [n for n in f["required"] if n not in key and n in present])
    out = []
    for n in ordered:
        item = format_value(n, float(present[n]))
        item["key"] = n in key
        out.append(item)
    return out


def present(report: dict, cancer_override=None) -> dict:
    status = report["status"]
    base = {"types": _types()}

    if status == "unreadable":
        return {**base, "status": "unreadable",
                "title": "We couldn't read this file",
                "message": ("Try a clearer photo with the page flat and well lit, "
                            "or upload the original PDF.")}

    if status == "rejected":
        return {**base, "status": "rejected",
                "title": "This doesn't look like a medical report",
                "message": ("Upload a lab, pathology or imaging report. Photos of other documents, "
                            "screenshots and unrelated PDFs can't be analysed.")}

    r = report["router"]
    cancer = r["chosen"]
    names = CANCER_NAMES[cancer]
    f = report["fields"]
    ocr = report["ocr"]
    low_ocr = ocr["confidence"] < config.LOW_OCR_CONFIDENCE and any(p.get("ocr") for p in ocr["pages"])

    common = {
        **base,
        "report": {"type": cancer, "label": names["label"], "condition": names["condition"],
                   "auto_detected": not r["override_used"]},
        "values": _values(report, cancer),
        "quality": {
            "ocr_confidence": ocr["confidence"],
            "low_ocr": bool(low_ocr),
            "values_read": len(f["required"]) - len(f["missing_all"]),
            "values_expected": len(f["required"]),
            "used_averages": len(f["missing_all"]) > 0,
        },
    }

    if f["missing_key"] and config.REQUIRE_KEY_FIELDS:
        return {**common, "status": "incomplete",
                "title": "We need a few more values",
                "message": ("This report is missing values the analysis relies on, so we can't "
                            "give a reliable result."),
                "missing": [label_for(n) for n in f["missing_key"]]}

    p = report["prediction"]
    band = BAND_KEYS[p["risk_band"]]
    score = p["final_score"]
    copy = _verdict_copy(band, names["label"], names["condition"])

    models = []
    for raw, prob in p["per_model"].items():
        short, kind = MODEL_NAMES.get(raw, (raw, "classical"))
        models.append({"name": short, "kind": kind, "flagged": prob >= 0.5, "prob": prob})

    stage = p["predicted_stage"]
    show_stage = stage is not None and stage > 0 and band != "low"

    return {
        **common,
        "status": "ok",
        "verdict": {"band": band, "score": round(score * 100), **copy},
        "early_risk": {"level": _level(p["early_risk_score"]),
                       "score": round(p["early_risk_score"] * 100)},
        "stage": {"value": stage, "label": f"Stage {ROMAN.get(stage, stage)}"} if show_stage else None,
        "agreement": {"flagged": sum(m["flagged"] for m in models), "total": len(models),
                      "models": models},
    }
