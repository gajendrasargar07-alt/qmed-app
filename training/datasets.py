"""
Dataset loaders for the five cancer types.

Each loader returns X, y (1 = positive), stage (0 to 4), feature_names, the
positive label and key_fields (the values a report must contain to be scored).

  breast      scikit-learn's Breast Cancer Wisconsin data: real, 569 patients.
  lung, prostate, colon, cervical
              simulated cohorts of 800 patients each. Marker distributions are
              set from published clinical ranges; these are not real patients.

Stages are generated from the simulated marker values. The seed is fixed.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer

RNG_SEED = 42
N_PER_CANCER = 800

# ------------------------------------------------------------------- breast
def load_breast():
    d = load_breast_cancer()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.int64)           # 0 = malignant, 1 = benign in sklearn
    y = 1 - y                                # flip -> 1 = malignant (positive)
    # stages: derive coarse stage from tumor size (mean radius) among malignant
    rng = np.random.RandomState(RNG_SEED)
    stage = np.zeros_like(y)
    for i in range(len(y)):
        if y[i] == 0:
            stage[i] = 0
        else:
            r = X[i, 0]                       # mean radius
            if r < 13:   stage[i] = 1
            elif r < 16: stage[i] = 2
            elif r < 20: stage[i] = 3
            else:        stage[i] = 4
    return dict(name="breast", X=X, y=y, stage=stage,
                feature_names=list(d.feature_names),
                positive="malignant tumor",
                key_fields=["mean radius", "mean texture", "mean perimeter",
                            "mean area", "mean concavity"])

# ------------------------------------------------------------------- lung
def load_lung(n=N_PER_CANCER, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.45).astype(np.int64)
    age            = np.clip(rng.normal(62 + 5*y, 9), 30, 90)
    pack_years     = np.clip(rng.normal(15 + 30*y, 12), 0, 90)
    fev1_pct       = np.clip(rng.normal(88 - 25*y, 12), 25, 120)
    cea            = np.clip(rng.lognormal(np.log(2.0 + 6*y), 0.5), 0.3, 200)
    cyfra          = np.clip(rng.lognormal(np.log(1.5 + 5*y), 0.6), 0.2, 150)
    nse            = np.clip(rng.lognormal(np.log(9 + 18*y), 0.4), 2, 200)
    nodule_mm      = np.clip(rng.normal(6 + 18*y, 5), 0, 60)
    spiculation    = np.clip(rng.normal(0.2 + 0.6*y, 0.2), 0, 1)
    ldh            = np.clip(rng.normal(210 + 90*y, 55), 100, 800)
    wbc            = np.clip(rng.normal(7.2 + 1.5*y, 2.1), 2, 25)
    hemoglobin     = np.clip(rng.normal(14 - 1.5*y, 1.3), 7, 18)
    platelets      = np.clip(rng.normal(260 + 40*y, 60), 80, 600)
    X = np.stack([age, pack_years, fev1_pct, cea, cyfra, nse, nodule_mm,
                  spiculation, ldh, wbc, hemoglobin, platelets], axis=1).astype(np.float32)
    stage = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if y[i] == 0: stage[i] = 0
        else:
            s = 0.3*nodule_mm[i]/10 + 0.4*(cea[i]>10) + 0.3*(cyfra[i]>3.3) + 0.3*(nse[i]>25)
            stage[i] = int(np.clip(np.round(1 + s), 1, 4))
    return dict(name="lung", X=X, y=y, stage=stage,
                feature_names=["age","pack_years","fev1_pct","cea","cyfra21_1","nse",
                               "nodule_mm","spiculation","ldh","wbc","hemoglobin","platelets"],
                positive="lung malignancy",
                key_fields=["cea","cyfra21_1","nse","nodule_mm","pack_years"])

# ------------------------------------------------------------------- prostate
def load_prostate(n=N_PER_CANCER, seed=RNG_SEED+1):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.45).astype(np.int64)
    age            = np.clip(rng.normal(63 + 4*y, 8), 40, 88)
    psa            = np.clip(rng.lognormal(np.log(2.5 + 10*y), 0.6), 0.1, 200)
    free_psa_ratio = np.clip(rng.normal(0.22 - 0.10*y, 0.06), 0.02, 0.5)
    prostate_vol   = np.clip(rng.normal(38 + 8*y, 15), 15, 150)
    dre_suspicious = (rng.rand(n) < (0.15 + 0.55*y)).astype(np.float32)
    gleason_proxy  = np.clip(rng.normal(3 + 3*y, 1.0), 0, 10)
    testosterone   = np.clip(rng.normal(500 - 40*y, 140), 100, 1100)
    alp            = np.clip(rng.normal(82 + 30*y, 25), 30, 400)
    hemoglobin     = np.clip(rng.normal(14.3 - 1.0*y, 1.3), 8, 18)
    creatinine     = np.clip(rng.normal(1.0 + 0.15*y, 0.25), 0.5, 4.0)
    X = np.stack([age, psa, free_psa_ratio, prostate_vol, dre_suspicious,
                  gleason_proxy, testosterone, alp, hemoglobin, creatinine], axis=1).astype(np.float32)
    stage = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if y[i] == 0: stage[i] = 0
        else:
            s = 0.4*(psa[i]>10) + 0.3*(psa[i]>20) + 0.3*dre_suspicious[i] + 0.3*(gleason_proxy[i]>7) + 0.2*(alp[i]>150)
            stage[i] = int(np.clip(np.round(1 + s), 1, 4))
    return dict(name="prostate", X=X, y=y, stage=stage,
                feature_names=["age","psa","free_psa_ratio","prostate_volume","dre_suspicious",
                               "gleason_proxy","testosterone","alp","hemoglobin","creatinine"],
                positive="prostate malignancy",
                key_fields=["psa","free_psa_ratio","gleason_proxy","alp"])

# ------------------------------------------------------------------- colon
def load_colon(n=N_PER_CANCER, seed=RNG_SEED+2):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.42).astype(np.int64)
    age          = np.clip(rng.normal(60 + 4*y, 10), 25, 90)
    cea          = np.clip(rng.lognormal(np.log(2.0 + 8*y), 0.6), 0.3, 300)
    ca19_9       = np.clip(rng.lognormal(np.log(15 + 25*y), 0.6), 1, 500)
    hemoglobin   = np.clip(rng.normal(13.8 - 2.0*y, 1.6), 6, 18)
    fit_test     = (rng.rand(n) < (0.05 + 0.7*y)).astype(np.float32)
    family_hx    = (rng.rand(n) < (0.10 + 0.15*y)).astype(np.float32)
    bmi          = np.clip(rng.normal(26 + 1.5*y, 4), 15, 45)
    smoker       = (rng.rand(n) < (0.15 + 0.10*y)).astype(np.float32)
    albumin      = np.clip(rng.normal(4.2 - 0.4*y, 0.4), 2, 5.5)
    crp          = np.clip(rng.lognormal(np.log(2 + 8*y), 0.7), 0.1, 200)
    X = np.stack([age, cea, ca19_9, hemoglobin, fit_test, family_hx, bmi,
                  smoker, albumin, crp], axis=1).astype(np.float32)
    stage = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if y[i] == 0: stage[i] = 0
        else:
            s = 0.3*(cea[i]>5) + 0.3*(cea[i]>15) + 0.2*(ca19_9[i]>37) + 0.3*(hemoglobin[i]<11) + 0.2*(crp[i]>10)
            stage[i] = int(np.clip(np.round(1 + s), 1, 4))
    return dict(name="colon", X=X, y=y, stage=stage,
                feature_names=["age","cea","ca19_9","hemoglobin","fit_test","family_history",
                               "bmi","smoker","albumin","crp"],
                positive="colorectal malignancy",
                key_fields=["cea","ca19_9","hemoglobin","fit_test"])

# ------------------------------------------------------------------- cervical
def load_cervical(n=N_PER_CANCER, seed=RNG_SEED+3):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < 0.35).astype(np.int64)
    age              = np.clip(rng.normal(38 + 6*y, 11), 18, 75)
    partners         = np.clip(rng.poisson(2 + 2*y), 0, 20)
    first_intercourse= np.clip(rng.normal(19 - 2*y, 3), 12, 35)
    pregnancies      = np.clip(rng.poisson(2 + y), 0, 12)
    smokes           = (rng.rand(n) < (0.15 + 0.20*y)).astype(np.float32)
    hormonal_yrs     = np.clip(rng.exponential(2 + 3*y), 0, 30)
    iud_yrs          = np.clip(rng.exponential(1 + 0.5*y), 0, 20)
    stds_count       = np.clip(rng.poisson(0.3 + 1.2*y), 0, 10)
    hpv_positive     = (rng.rand(n) < (0.10 + 0.70*y)).astype(np.float32)
    dx_cin           = (rng.rand(n) < (0.02 + 0.55*y)).astype(np.float32)
    schiller         = (rng.rand(n) < (0.05 + 0.60*y)).astype(np.float32)
    citology         = (rng.rand(n) < (0.03 + 0.65*y)).astype(np.float32)
    X = np.stack([age, partners, first_intercourse, pregnancies, smokes,
                  hormonal_yrs, iud_yrs, stds_count, hpv_positive, dx_cin,
                  schiller, citology], axis=1).astype(np.float32)
    stage = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if y[i] == 0: stage[i] = 0
        else:
            s = 0.4*hpv_positive[i] + 0.3*dx_cin[i] + 0.3*schiller[i] + 0.2*citology[i] + 0.2*(stds_count[i]>2)
            stage[i] = int(np.clip(np.round(1 + s), 1, 4))
    return dict(name="cervical", X=X, y=y, stage=stage,
                feature_names=["age","partners","first_intercourse_age","pregnancies","smokes",
                               "hormonal_contraceptives_years","iud_years","stds_count",
                               "hpv_positive","dx_cin","schiller","citology"],
                positive="cervical malignancy",
                key_fields=["hpv_positive","citology","schiller","dx_cin","stds_count"])

# ------------------------------------------------------------------- registry
LOADERS = {
    "breast":   load_breast,
    "lung":     load_lung,
    "prostate": load_prostate,
    "colon":    load_colon,
    "cervical": load_cervical,
}

def load_all():
    return {k: fn() for k, fn in LOADERS.items()}

if __name__ == "__main__":
    for k, ds in load_all().items():
        print(f"{k:10s} X={ds['X'].shape} y+={int(ds['y'].sum())} stages={np.bincount(ds['stage']).tolist()}")
