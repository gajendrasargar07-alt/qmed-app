"""upload -> OCR -> validate -> route -> extract values -> score."""
import joblib
import numpy as np

from . import config
from .ocr import extract, extract_fields, flag_missing
from .scoring import load_bundle, score_from_fields
from .validator import validate

# Report wording -> the feature name a model was trained with.
FIELD_ALIAS = {
    "gleason":        ["gleason_proxy", "gleason"],
    "ca19_9":         ["ca19_9"],
    "cyfra21_1":      ["cyfra21_1"],
    "free_psa_ratio": ["free_psa_ratio"],
    "nodule_mm":      ["nodule_mm"],
    "fev1_pct":       ["fev1_pct"],
    "pack_years":     ["pack_years"],
    "prostate_volume": ["prostate_volume"],
    "dre_suspicious": ["dre_suspicious"],
    "hpv_positive":   ["hpv_positive"],
    "dx_cin":         ["dx_cin"],
    "schiller":       ["schiller"],
    "citology":       ["citology"],
    "fit_test":       ["fit_test"],
    "spiculation":    ["spiculation"],
    "smoker":         ["smokes"],
}

_router = None


def _load_router():
    global _router
    if _router is None:
        _router = joblib.load(config.ROUTER_PATH)
    return _router


def _apply_aliases(fields: dict, feature_names) -> dict:
    out = dict(fields)
    for src, targets in FIELD_ALIAS.items():
        if src in out:
            for t in targets:
                if t in feature_names and t not in out:
                    out[t] = out[src]
    return out


def analyze(source, source_kind: str = "auto", cancer: str = None) -> dict:
    ocr = extract(source, source_kind)
    text = ocr["text"]

    if ocr["source_kind"] != "text" and len(text.strip()) < 25:
        return {"status": "unreadable", "ocr": ocr}

    v = validate(text)
    if not v["is_medical"]:
        return {"status": "rejected", "ocr": ocr, "validation": v}

    pipe = _load_router()["pipeline"]
    probs = pipe.predict_proba([text])[0]
    ranked = sorted(zip(pipe.classes_, probs), key=lambda cp: -cp[1])
    predicted = str(ranked[0][0])
    chosen = cancer or v.get("cancer_hint") or predicted

    bundle = load_bundle(chosen)
    required = [str(n) for n in bundle["feature_names"]]
    key_fields = [str(n) for n in bundle["key_fields"]]
    fields = _apply_aliases(extract_fields(text), required)

    missing_all = flag_missing(fields, required)
    missing_key = flag_missing(fields, key_fields)
    prediction = score_from_fields(chosen, fields)

    return {
        "status": "ok",
        "ocr": ocr,
        "validation": v,
        "router": {
            "chosen": chosen,
            "predicted": predicted,
            "override_used": bool(cancer),
            "distribution": [{"cancer": str(c), "prob": round(float(p), 4)} for c, p in ranked],
        },
        "fields": {
            "extracted": {k: (round(float(x), 4) if isinstance(x, (int, float, np.floating)) else x)
                          for k, x in fields.items()},
            "required": required,
            "key_fields": key_fields,
            "completeness_all": missing_all["completeness"],
            "completeness_key": missing_key["completeness"],
            "missing_all": missing_all["missing"],
            "missing_key": missing_key["missing"],
        },
        "prediction": prediction,
    }
