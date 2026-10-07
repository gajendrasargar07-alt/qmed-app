"""
Train a text-based cancer-type router.

Given the raw text of a medical report (or extracted lab-field labels), the
router predicts which cancer domain to route it to. Uses TF-IDF word/char
n-grams + Logistic Regression. Training corpus is a curated set of medical
keywords / phrasings per cancer plus synthetic report sentences.
"""
import os, joblib, numpy as np, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "router", "router.joblib")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

CORPUS = {
    "breast": [
        "fine needle aspiration cytology of breast lump",
        "BRCA1 BRCA2 mutation panel positive",
        "HER2 ER PR status ki-67 index",
        "mean radius mean texture mean perimeter concavity mammography",
        "breast mass biopsy report malignant infiltrating ductal carcinoma",
        "CA 15-3 elevated bilateral mammogram BIRADS 4",
        "mean smoothness compactness fractal dimension nucleus features",
        "breast ultrasound irregular hypoechoic lesion",
    ],
    "lung": [
        "chest CT nodule spiculated margins pack years",
        "CEA CYFRA 21-1 NSE tumor markers lung",
        "pulmonary function test FEV1 forced expiratory volume",
        "PET CT SUV lung mass mediastinal lymph nodes",
        "adenocarcinoma non small cell lung cancer NSCLC biopsy",
        "cough hemoptysis dyspnea history smoker",
        "solitary pulmonary nodule ground glass opacity",
        "EGFR ALK ROS1 lung mutation panel",
    ],
    "prostate": [
        "PSA prostate specific antigen free total ratio",
        "digital rectal examination DRE gleason score",
        "prostate biopsy transrectal ultrasound",
        "prostate volume MRI PI-RADS score",
        "adenocarcinoma prostate metastatic bone scan",
        "testosterone alkaline phosphatase prostate",
        "urinary frequency nocturia BPH prostate",
        "gleason 3+4 7 prostatectomy specimen",
    ],
    "colon": [
        "colonoscopy polyp adenoma colorectal",
        "CEA CA 19-9 colorectal cancer markers",
        "fecal immunochemical test FIT positive",
        "hemoglobin anemia rectal bleeding",
        "sigmoidoscopy biopsy adenocarcinoma colon",
        "family history colorectal Lynch syndrome",
        "CT abdomen liver metastases colorectal",
        "polypectomy tubular adenoma high grade dysplasia",
    ],
    "cervical": [
        "pap smear cytology HPV positive",
        "cervical intraepithelial neoplasia CIN 2 CIN 3",
        "colposcopy Schiller test acetowhite epithelium",
        "human papillomavirus 16 18 high risk",
        "cervical biopsy squamous cell carcinoma",
        "sexual partners contraceptive history STD",
        "LEEP conization cervical dysplasia",
        "cervix pap Bethesda ASCUS LSIL HSIL",
    ],
}

def build_dataset(seed=42):
    rng = random.Random(seed)
    texts, labels = [], []
    for label, phrases in CORPUS.items():
        # augment: random combos of 2-3 phrases + boilerplate
        for p in phrases:
            texts.append(p); labels.append(label)
        for _ in range(60):
            k = rng.randint(2, 4)
            combo = " . ".join(rng.sample(phrases, k=min(k, len(phrases))))
            preamble = rng.choice([
                "Patient report:", "Clinical summary:", "Lab findings:",
                "Radiology impression:", "Pathology report:", "History:",
            ])
            texts.append(f"{preamble} {combo}")
            labels.append(label)
    return texts, labels

def main():
    X, y = build_dataset()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=8000,
                                   sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")),
    ])
    pipe.fit(X, y)
    train_acc = pipe.score(X, y)
    print(f"Router training accuracy: {train_acc:.4f}  (n={len(X)}, classes={sorted(set(y))})")
    joblib.dump({"pipeline": pipe, "classes": sorted(set(y))}, OUT, compress=3)
    print(f"saved -> {OUT}")

if __name__ == "__main__":
    main()
