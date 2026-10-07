"""Checks whether extracted text looks like a medical report."""
import re

MEDICAL_LEXICON = {
    # general clinical
    "patient", "report", "clinical", "diagnosis", "history", "impression",
    "findings", "specimen", "biopsy", "pathology", "laboratory", "lab",
    "radiology", "imaging", "ultrasound", "mri", "ct", "pet", "xray",
    "hospital", "physician", "referred", "sample", "collected",
    # oncology
    "cancer", "tumor", "tumour", "malignant", "benign", "carcinoma",
    "adenocarcinoma", "sarcoma", "lymphoma", "leukemia", "metastasis",
    "metastatic", "oncology", "neoplasm", "dysplasia", "stage",
    # markers and labs
    "brca1", "brca2", "her2", "ki-67", "ki67", "er", "pr", "psa", "cea",
    "ca19-9", "ca15-3", "cyfra", "nse", "afp", "hcg", "ldh", "alp",
    "hemoglobin", "wbc", "platelets", "creatinine", "albumin", "crp",
    "testosterone", "hpv", "cin", "gleason", "fev1", "cyfra21-1",
    "birads", "pi-rads",
    # breast cytology features
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave", "symmetry", "fractal",
    # cervical
    "pap", "smear", "cytology", "schiller", "iud", "hormonal",
    # colon
    "colonoscopy", "polyp", "adenoma", "colorectal", "fit", "fecal",
    # lung
    "nodule", "spiculated", "pack-years", "pack", "years", "smoker",
    # molecular and generic result vocabulary
    "mutation", "gene", "genotype", "snp", "rs", "allele", "expression",
    "positive", "negative", "elevated", "decreased", "normal",
    "grade", "score", "level", "count", "mg", "ml", "dl", "mm",
    "biomarker", "marker", "panel",
}

UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ng/ml|mg/dl|g/dl|u/ml|iu/ml|meq/l|mmhg|"
    r"cells/[uμ]l|/mm3|mm\^?3|ml|mm|cm|kg|%|units?|iu)\b",
    re.IGNORECASE,
)

CANCER_HINTS = {
    "breast":   ["breast", "brca", "her2", "mammogram", "birads", "ductal"],
    "lung":     ["lung", "pulmonary", "chest ct", "nodule", "smoker", "fev1", "cyfra", "nsclc"],
    "prostate": ["prostate", "psa", "gleason", "dre", "transrectal"],
    "colon":    ["colon", "colorectal", "cea", "ca19-9", "colonoscopy", "polyp", "fit test"],
    "cervical": ["cervix", "cervical", "pap", "hpv", "schiller", "cin", "colposcopy"],
}

ANCHOR_TERMS = ("cancer", "tumor", "tumour", "malignant", "carcinoma",
                "oncology", "biopsy", "psa", "brca", "hpv", "pap smear")


def _tokens(text):
    return [t.lower() for t in re.findall(r"[A-Za-z0-9\-]+", text)]


def validate(text: str, min_score: float = 0.18) -> dict:
    if not text or len(text.strip()) < 25:
        return {
            "is_medical": False, "score": 0.0, "matched_terms": [],
            "matched_units": 0, "cancer_hint": None,
            "reason": "Not enough text to be a medical report.",
        }

    toks = _tokens(text)
    matched = [t for t in toks if t in MEDICAL_LEXICON]
    unique = sorted(set(matched))
    med_density = len(matched) / max(len(toks), 1)
    units = len(UNIT_PATTERN.findall(text))
    unit_density = min(units / 20.0, 1.0)

    score = 0.65 * med_density + 0.35 * unit_density
    low = text.lower()
    if any(k in low for k in ANCHOR_TERMS):
        score = min(score + 0.08, 1.0)

    is_medical = score >= min_score and len(unique) >= 4

    hint, best = None, 0
    for cancer, words in CANCER_HINTS.items():
        n = sum(1 for w in words if w in low)
        if n > best:
            best, hint = n, cancer

    return {
        "is_medical": bool(is_medical),
        "score": round(float(score), 4),
        "matched_terms": unique[:40],
        "matched_units": int(units),
        "cancer_hint": hint if best >= 1 else None,
        "reason": f"Medical density {med_density:.2%}, {units} unit hits, {len(unique)} terms matched.",
    }
