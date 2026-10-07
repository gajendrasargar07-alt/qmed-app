"""Plain-English names, units and display style for every value the models use."""

# kind: num = number, pos = Positive/Negative, yes = Yes/No,
#       susp = Suspicious/Not suspicious, abn = Abnormal/Normal
SPECS = {
    # shared
    "age":          ("Age", "years", "num"),
    "hemoglobin":   ("Hemoglobin", "g/dL", "num"),
    "wbc":          ("White blood cells", "x10⁹/L", "num"),
    "platelets":    ("Platelets", "x10⁹/L", "num"),
    "creatinine":   ("Creatinine", "mg/dL", "num"),
    "albumin":      ("Albumin", "g/dL", "num"),
    "crp":          ("C-reactive protein", "mg/L", "num"),
    "bmi":          ("BMI", "", "num"),
    "ldh":          ("LDH", "U/L", "num"),
    "alp":          ("Alkaline phosphatase", "U/L", "num"),
    "cea":          ("CEA", "ng/mL", "num"),
    # prostate
    "psa":              ("PSA", "ng/mL", "num"),
    "free_psa_ratio":   ("Free/total PSA ratio", "", "num"),
    "prostate_volume":  ("Prostate volume", "mL", "num"),
    "dre_suspicious":   ("Digital rectal exam", "", "susp"),
    "gleason_proxy":    ("Gleason score", "", "num"),
    "testosterone":     ("Testosterone", "ng/dL", "num"),
    # lung
    "pack_years":   ("Smoking history", "pack-years", "num"),
    "fev1_pct":     ("Lung function (FEV1)", "%", "num"),
    "cyfra21_1":    ("CYFRA 21-1", "ng/mL", "num"),
    "nse":          ("NSE", "ng/mL", "num"),
    "nodule_mm":    ("Nodule size", "mm", "num"),
    "spiculation":  ("Spiculated edges", "", "yes"),
    # colon
    "ca19_9":         ("CA 19-9", "U/mL", "num"),
    "fit_test":       ("FIT stool test", "", "pos"),
    "family_history": ("Family history", "", "yes"),
    "smoker":         ("Smoker", "", "yes"),
    # cervical
    "partners":                       ("Sexual partners", "", "num"),
    "first_intercourse_age":          ("Age at first intercourse", "years", "num"),
    "pregnancies":                    ("Pregnancies", "", "num"),
    "smokes":                         ("Smoker", "", "yes"),
    "hormonal_contraceptives_years":  ("Hormonal contraceptive use", "years", "num"),
    "iud_years":                      ("IUD use", "years", "num"),
    "stds_count":                     ("Past STDs", "", "num"),
    "hpv_positive":                   ("HPV test", "", "pos"),
    "dx_cin":                         ("CIN found on colposcopy", "", "yes"),
    "schiller":                       ("Schiller test", "", "pos"),
    "citology":                       ("Pap smear cytology", "", "abn"),
}

CANCER_NAMES = {
    "breast":   {"label": "Breast",   "condition": "breast cancer"},
    "lung":     {"label": "Lung",     "condition": "lung cancer"},
    "prostate": {"label": "Prostate", "condition": "prostate cancer"},
    "colon":    {"label": "Colon",    "condition": "colorectal cancer"},
    "cervical": {"label": "Cervical", "condition": "cervical cancer"},
}

MODEL_NAMES = {
    "Logistic Regression":            ("Logistic regression", "classical"),
    "Random Forest":                  ("Random forest", "classical"),
    "Gradient Boosting":              ("Gradient boosting", "classical"),
    "Classical SVM (RBF)":            ("Support vector machine", "classical"),
    "Quantum Kernel SVM":             ("Quantum kernel", "quantum"),
    "Variational Quantum Classifier": ("Variational quantum", "quantum"),
}

_WORDS = {
    "pos":  ("Positive", "Negative"),
    "yes":  ("Yes", "No"),
    "susp": ("Suspicious", "Not suspicious"),
    "abn":  ("Abnormal", "Normal"),
}


def spec_for(name: str):
    if name in SPECS:
        return SPECS[name]
    # breast cytology features: "mean radius", "radius error", "worst texture"
    return (name[:1].upper() + name[1:], "", "num")


def label_for(name: str) -> str:
    return spec_for(name)[0]


def format_value(name: str, value: float) -> dict:
    label, unit, kind = spec_for(name)
    if kind in _WORDS:
        text = _WORDS[kind][0 if value >= 0.5 else 1]
        unit = ""
    else:
        text = f"{value:.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return {"name": name, "label": label, "value": text, "unit": unit}
