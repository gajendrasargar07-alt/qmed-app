from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
CANCER_MODELS_DIR = MODELS_DIR / "multicancer"
ROUTER_PATH = MODELS_DIR / "router" / "router.joblib"

CANCERS = ("breast", "lung", "prostate", "colon", "cervical")

MAX_UPLOAD_MB = 15
MAX_PDF_PAGES = 10
OCR_TIMEOUT_S = 60

# Risk bands on the unified score.
HIGH_THRESHOLD = 0.75
MODERATE_THRESHOLD = 0.40

# When True, a report that lacks any of a cancer type's key values gets no risk
# verdict (the missing values are listed instead). Missing values are otherwise
# filled with training-set averages, which can hide a real problem.
REQUIRE_KEY_FIELDS = True

# Below this OCR confidence we tell the person to double-check what was read.
LOW_OCR_CONFIDENCE = 0.55
