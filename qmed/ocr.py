"""Turns an uploaded report (text, PDF or image) into text, then pulls out lab values."""
import io
import os
import re

import numpy as np

from . import config

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    import pytesseract
    from pytesseract import Output
except Exception:  # pragma: no cover
    pytesseract = None
try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None
try:
    from pdf2image import convert_from_bytes
except Exception:  # pragma: no cover
    convert_from_bytes = None
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def tesseract_available() -> bool:
    if pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# Spellings that OCR commonly mangles, mapped to the form the parser expects.
MED_VOCAB = {
    "psa": ["p.s.a.", "p s a", "psa"],
    "cea": ["c.e.a.", "c e a", "cea"],
    "ca 15-3": ["ca15-3", "ca 15 3", "ca-15-3", "ca 153"],
    "ca 19-9": ["ca19-9", "ca 19 9", "ca-19-9", "ca 199"],
    "her2": ["her 2", "her-2", "her/2"],
    "brca1": ["brca 1", "brca-1"],
    "brca2": ["brca 2", "brca-2"],
    "ki-67": ["ki 67", "ki67", "ki-67"],
    "gleason": ["gleeson", "glcason", "gieason"],
    "hemoglobin": ["haemoglobin", "hemogloibin"],
    "hpv": ["h.p.v", "h p v"],
    "cyfra 21-1": ["cyfra21-1", "cyfra 211"],
    "fev1": ["fev-1", "fev 1", "fev1%"],
}


def _correct_medical_terms(text: str) -> str:
    for canonical, variants in MED_VOCAB.items():
        for v in variants:
            text = re.sub(re.escape(v), canonical, text, flags=re.IGNORECASE)
    # digit/letter confusions between numbers
    text = re.sub(r"(?<=\d)[Oo](?=\d)", "0", text)
    text = re.sub(r"(?<=\d)[lI](?=\d)", "1", text)
    return text


def _preprocess_image(img_bgr):
    if cv2 is None:
        return img_bgr
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def _ocr_image(pil_img):
    """Returns (text, mean confidence 0..1)."""
    if pytesseract is None:
        return "", 0.0
    if cv2 is not None:
        arr = np.array(pil_img.convert("RGB"))[..., ::-1]  # RGB -> BGR
        img = Image.fromarray(_preprocess_image(arr))
    else:
        img = pil_img
    cfg = "--psm 6 -c preserve_interword_spaces=1"
    text = pytesseract.image_to_string(img, config=cfg, timeout=config.OCR_TIMEOUT_S)
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config=cfg,
                                     timeout=config.OCR_TIMEOUT_S)
    confs = [int(c) for c in data.get("conf", [])
             if str(c).lstrip("-").isdigit() and int(c) > 0]
    return text, (float(np.mean(confs)) / 100 if confs else 0.0)


def sniff_kind(data: bytes) -> str:
    """Decide what an upload is from its first bytes: 'pdf', 'image' or 'text'."""
    head = data[:12]
    if head.startswith(b"%PDF"):
        return "pdf"
    if (head.startswith(b"\x89PNG") or head[:3] == b"\xff\xd8\xff"
            or head[:4] in (b"II*\x00", b"MM\x00*") or head[:2] == b"BM"
            or (head[:4] == b"RIFF" and data[8:12] == b"WEBP")):
        return "image"
    return "text"


def extract(source, source_kind: str = "auto") -> dict:
    """
    source: bytes, a file path, or a plain string.
    Returns text, confidence, pages, pipeline, warnings and source_kind.
    """
    warnings, pages, pipeline = [], [], []
    text, conf = "", 1.0

    if source_kind == "auto":
        if isinstance(source, (bytes, bytearray)):
            source_kind = sniff_kind(bytes(source))
        elif isinstance(source, str) and os.path.exists(source):
            ext = os.path.splitext(source)[1].lower()
            source_kind = ("pdf" if ext == ".pdf" else
                           "image" if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
                           else "text")
        else:
            source_kind = "text"

    if source_kind == "text":
        if isinstance(source, (bytes, bytearray)):
            text = bytes(source).decode("utf-8", errors="ignore")
        elif os.path.exists(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as fp:
                text = fp.read()
        else:
            text = str(source)
        pipeline.append("plain-text")

    elif source_kind == "pdf":
        pdf_bytes = bytes(source) if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
        if pdfplumber is not None:
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    chunks = []
                    for page in pdf.pages[:config.MAX_PDF_PAGES]:
                        t = page.extract_text() or ""
                        chunks.append(t)
                        pages.append({"page": len(pages) + 1, "text_len": len(t), "ocr": False})
                    text = "\n\n".join(chunks)
                pipeline.append("pdf-text-layer")
            except Exception as e:
                warnings.append(f"pdf text layer failed: {e}")
        if len(text.strip()) < 60:  # scanned PDF: rasterize and OCR
            pipeline.append("pdf-rasterize+tesseract")
            pages = []
            if convert_from_bytes is None or pytesseract is None:
                warnings.append("OCR is not available, so scanned PDFs cannot be read.")
            else:
                try:
                    imgs = convert_from_bytes(pdf_bytes, dpi=220, first_page=1,
                                              last_page=config.MAX_PDF_PAGES)
                except Exception as e:
                    warnings.append(f"pdf rasterize failed: {e}")
                    imgs = []
                texts, confs = [], []
                for i, im in enumerate(imgs):
                    t, c = _ocr_image(im)
                    texts.append(t)
                    confs.append(c)
                    pages.append({"page": i + 1, "text_len": len(t), "ocr": True,
                                  "confidence": round(c, 3)})
                text = "\n\n".join(texts)
                conf = float(np.mean(confs)) if confs else 0.0

    elif source_kind == "image":
        img = Image.open(io.BytesIO(bytes(source))) if isinstance(source, (bytes, bytearray)) else Image.open(source)
        if pytesseract is None:
            warnings.append("OCR is not available, so images cannot be read.")
        t, c = _ocr_image(img)
        text, conf = t, c
        pipeline.append("image+tesseract")
        pages.append({"page": 1, "text_len": len(t), "ocr": True, "confidence": round(c, 3)})

    else:
        raise ValueError(f"Unsupported source_kind: {source_kind}")

    if conf < config.LOW_OCR_CONFIDENCE and any(p.get("ocr") for p in pages):
        warnings.append(f"Low OCR confidence ({conf:.2f}).")

    return {
        "text": _correct_medical_terms(text),
        "raw_text": text,
        "confidence": round(float(conf), 4),
        "pages": pages,
        "pipeline": pipeline,
        "warnings": warnings,
        "source_kind": source_kind,
    }


# ---------------------------------------------------------------- field extraction
NUM = r"(-?\d+(?:\.\d+)?)"

FIELD_PATTERNS = {
    "age":          [r"\bage\s*[:=]?\s*" + NUM, r"\b" + NUM + r"\s*[- ]?year[- ]?old"],
    "hemoglobin":   [r"h[ae]moglobin\s*[:=]?\s*" + NUM, r"\bhb\b\s*[:=]?\s*" + NUM],
    "wbc":          [r"\bwbc\s*[:=]?\s*" + NUM, r"\bleukocytes?\s*[:=]?\s*" + NUM],
    "platelets":    [r"platelets?\s*[:=]?\s*" + NUM],
    "creatinine":   [r"creatinine\s*[:=]?\s*" + NUM],
    "albumin":      [r"albumin\s*[:=]?\s*" + NUM],
    "crp":          [r"\bcrp\s*[:=]?\s*" + NUM, r"c[- ]reactive protein\s*[:=]?\s*" + NUM],
    "bmi":          [r"\bbmi\s*[:=]?\s*" + NUM],
    "ldh":          [r"\bldh\s*[:=]?\s*" + NUM],
    "alp":          [r"\balp\s*[:=]?\s*" + NUM, r"alkaline phosphatase\s*[:=]?\s*" + NUM],
    "psa":          [r"\bpsa\s*[:=]?\s*" + NUM],
    "free_psa_ratio": [r"free\s*/?\s*total\s*psa\s*(?:ratio)?\s*[:=]?\s*" + NUM,
                       r"free\s*psa\s*ratio\s*[:=]?\s*" + NUM],
    "cea":          [r"\bcea\s*[:=]?\s*" + NUM],
    "ca19_9":       [r"ca\s*19[- ]?9\s*[:=]?\s*" + NUM],
    "ca15_3":       [r"ca\s*15[- ]?3\s*[:=]?\s*" + NUM],
    "cyfra21_1":    [r"cyfra\s*21[- ]?1\s*[:=]?\s*" + NUM],
    "nse":          [r"\bnse\s*[:=]?\s*" + NUM],
    "gleason":      [r"gleason(?:\s*score)?\s*[:=]?\s*" + NUM],
    "ki_67":        [r"ki[- ]?67\s*[:=]?\s*" + NUM],
    "testosterone": [r"testosterone\s*[:=]?\s*" + NUM],
    "fev1_pct":     [r"fev1\s*(?:%)?\s*[:=]?\s*" + NUM],
    "pack_years":   [r"pack[- ]?years?\s*[:=]?\s*" + NUM],
    "nodule_mm":    [r"nodule\s*(?:size)?\s*[:=]?\s*" + NUM + r"\s*mm"],
    "prostate_volume": [r"prostate\s*volume\s*[:=]?\s*" + NUM],
    "partners":     [r"(?:sexual\s*)?partners\s*[:=]?\s*" + NUM],
    "pregnancies":  [r"pregnanc(?:y|ies)\s*[:=]?\s*" + NUM],
    "stds_count":   [r"stds?\s*(?:count)?\s*[:=]?\s*" + NUM],
    "first_intercourse_age": [r"first\s*intercourse\s*(?:age)?\s*[:=]?\s*" + NUM],
    "iud_years":    [r"iud\s*(?:\(?years\)?)?\s*[:=]?\s*" + NUM],
    "hormonal_contraceptives_years": [r"hormonal.*(?:years)?\s*[:=]?\s*" + NUM],
}

# Breast cytology panel: 10 base measurements x (mean, error, worst).
_BREAST_BASES = {
    "radius": r"radius", "texture": r"texture", "perimeter": r"perimeter",
    "area": r"area", "smoothness": r"smoothness", "compactness": r"compactness",
    "concavity": r"concavity", "concave points": r"concave\s+points",
    "symmetry": r"symmetry", "fractal dimension": r"fractal\s+dimension",
}
for _name, _rx in _BREAST_BASES.items():
    FIELD_PATTERNS[f"mean {_name}"] = [r"\bmean\s+" + _rx + r"\s*[:=]?\s*" + NUM]
    FIELD_PATTERNS[f"{_name} error"] = [r"\b" + _rx + r"\s+error\s*[:=]?\s*" + NUM]
    FIELD_PATTERNS[f"worst {_name}"] = [r"\bworst\s+" + _rx + r"\s*[:=]?\s*" + NUM]

# name -> (regex capturing the answer, words that count as a positive answer)
BOOL_PATTERNS = {
    "smoker":         (r"smok(?:er|es|ing)\s*[:=]?\s*(yes|no|pos|neg|positive|negative|\d+)", ("yes", "pos", "positive")),
    "family_history": (r"family\s*(?:history|hx).{0,30}?(yes|no|pos|neg|positive|negative)", ("yes", "pos", "positive")),
    "dre_suspicious": (r"dre\s*[:=]?\s*(suspicious|normal|abnormal|pos|neg|positive|negative)", ("suspicious", "abnormal", "pos", "positive")),
    "hpv_positive":   (r"hpv(?:\s+test(?:ing)?)?\s*[:=]?\s*(positive|negative|pos|neg|yes|no|high[- ]risk)", ("positive", "pos", "yes", "high-risk", "high risk")),
    "dx_cin":         (r"cin\s*[:=]?\s*(?:grade)?\s*(\d|none|no|neg|negative)", ("1", "2", "3")),
    "schiller":       (r"schiller(?:\s+test)?\s*[:=]?\s*(pos|neg|positive|negative)", ("positive", "pos")),
    "citology":       (r"c[iy]tology\s*[:=]?\s*(pos|neg|positive|negative|normal|abnormal)", ("positive", "pos", "abnormal")),
    "fit_test":       (r"fit\s*(?:test)?\s*[:=]?\s*(pos|neg|positive|negative)", ("positive", "pos")),
    "spiculation":    (r"spiculat(?:ion|ed)\s*[:=]?\s*(pos|neg|positive|negative|yes|no)", ("positive", "pos", "yes")),
}


def extract_fields(text: str) -> dict:
    fields = {}
    low = text.lower()
    for name, patterns in FIELD_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, low, flags=re.IGNORECASE)
            if m:
                try:
                    fields[name] = float(m.group(1))
                except ValueError:
                    pass
                break
    for name, (pat, positives) in BOOL_PATTERNS.items():
        m = re.search(pat, low, flags=re.IGNORECASE)
        if m:
            v = m.group(1).lower()
            fields[name] = float(any(p in v for p in positives) or (v.isdigit() and int(v) > 0))
    return fields


def flag_missing(fields: dict, required) -> dict:
    required = list(required)
    missing = [r for r in required if r not in fields]
    return {
        "missing": missing,
        "present": [r for r in required if r in fields],
        "completeness": round(1 - len(missing) / max(len(required), 1), 3),
    }
