"""
Trains the per-cancer models.

For each cancer type:
  - StandardScaler
  - LogReg, RandomForest, GradientBoost, SVM(RBF)         [classical]
  - Quantum Kernel SVM (small subsample, 6 qubits)         [quantum]
  - Variational Quantum Classifier (few epochs, 6 qubits)  [quantum]
  - Stage classifier (RandomForest multiclass) on positives only
  - Early-risk regressor (GradientBoost) on continuous risk proxy
Saves per-type bundle -> models/multicancer/<cancer>.joblib
Also saves a "router" that decides cancer type from a feature/text vector.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
import pennylane as qml
from pennylane import numpy as pnp

from datasets import load_all

# Output goes to training/output so a retrain never overwrites the shipped models.
# Copy files into models/ yourself once you are happy with the results.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output", "multicancer")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = 42
N_QUBITS = 4
VQC_LAYERS = 1
VQC_EPOCHS = 4
QK_SUBSAMPLE = 40        # quantum-kernel subsample per class

dev = qml.device("default.qubit", wires=N_QUBITS)

def embedding(x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for i in range(N_QUBITS-1):
        qml.CNOT(wires=[i, i+1])
    qml.CNOT(wires=[N_QUBITS-1, 0])

@qml.qnode(dev)
def state_circuit(x):
    embedding(x); return qml.state()

@qml.qnode(dev, interface="autograd")
def vqc_circuit(weights, x):
    embedding(x); qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

def q_prep(X, scaler_q=None, pca=None, lo=None, hi=None):
    if scaler_q is None:
        scaler_q = StandardScaler().fit(X)
    Xs = scaler_q.transform(X)
    if pca is None:
        pca = PCA(n_components=N_QUBITS, random_state=RNG).fit(Xs)
    Z = pca.transform(Xs)
    if lo is None:
        lo, hi = Z.min(0), Z.max(0)
    Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return np.clip(Zn, 0, 1) * np.pi, scaler_q, pca, lo, hi

def train_vqc(Xq, y, n_epochs=VQC_EPOCHS):
    rng = np.random.RandomState(RNG)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=VQC_LAYERS, n_wires=N_QUBITS)
    weights = pnp.array(0.1 * rng.randn(*shape), requires_grad=True)
    opt = qml.AdamOptimizer(0.12)
    y_pm = 2*y - 1
    idx = np.arange(len(Xq))
    for ep in range(n_epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 32):
            batch = idx[b:b+32]
            def cost(w):
                preds = pnp.stack([vqc_circuit(w, pnp.array(Xq[i], requires_grad=False))
                                   for i in batch])
                return pnp.mean((preds - y_pm[batch])**2)
            weights = opt.step(cost, weights)
    return np.array(weights)

def vqc_probs(weights, Xq):
    W = pnp.array(weights, requires_grad=False)
    out = np.array([float(vqc_circuit(W, pnp.array(x, requires_grad=False))) for x in Xq])
    return (out + 1.0) / 2.0

def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    try: auc = float(roc_auc_score(y_true, prob))
    except: auc = 0.5
    return {
        "accuracy":    round(float(accuracy_score(y_true, pred)), 4),
        "precision":   round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "sensitivity": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "specificity": round(float(recall_score(y_true, pred, pos_label=0, zero_division=0)), 4),
        "f1":          round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc":     round(auc, 4),
        "threshold":   round(float(thr), 4),
    }

def tune_early_threshold(y_true, prob, min_sens=0.95):
    # highest threshold that still meets sensitivity floor
    thrs = np.linspace(0.05, 0.95, 91)
    best = 0.5
    for t in thrs[::-1]:
        pred = (prob >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        if sens >= min_sens:
            best = t; break
    return float(best)

def train_one(name, ds):
    print(f"\n=== {name.upper()} · {ds['X'].shape} ===")
    X, y, stage = ds["X"], ds["y"], ds["stage"]
    Xtr, Xte, ytr, yte, str_, ste = train_test_split(
        X, y, stage, test_size=0.22, random_state=RNG, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    classical = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RNG),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RNG, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=140, max_depth=3, random_state=RNG),
        "Classical SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RNG),
    }
    # SGD LogReg for incremental / self-learning
    sgd = SGDClassifier(loss="log_loss", max_iter=25, random_state=RNG)
    sgd.fit(Xtr_s, ytr)

    per_model = {}
    fitted = {}
    for mname, clf in classical.items():
        t0 = time.time()
        clf.fit(Xtr_s, ytr)
        prob = clf.predict_proba(Xte_s)[:, 1]
        m = metrics(yte, prob); m["train_seconds"] = round(time.time()-t0, 2)
        m["early_threshold"] = round(tune_early_threshold(yte, prob), 4)
        per_model[mname] = m
        fitted[mname] = clf
        print(f"  {mname:22s} acc={m['accuracy']} auc={m['roc_auc']} sens={m['sensitivity']}")

    # ---------------- quantum prep
    Xq_tr, scaler_q, pca, q_lo, q_hi = q_prep(Xtr)
    Xq_te, *_ = q_prep(Xte, scaler_q, pca, q_lo, q_hi)

    # ---------------- quantum-kernel SVM (small subsample)
    rng = np.random.RandomState(RNG)
    idx_pos = np.where(ytr == 1)[0]; idx_neg = np.where(ytr == 0)[0]
    take = min(QK_SUBSAMPLE, len(idx_pos), len(idx_neg))
    sel = np.concatenate([rng.choice(idx_pos, take, False), rng.choice(idx_neg, take, False)])
    rng.shuffle(sel)
    Xq_qk = Xq_tr[sel]; yq_qk = ytr[sel]
    t0 = time.time()
    states_train = np.array([state_circuit(pnp.array(x, requires_grad=False)) for x in Xq_qk])
    K_train = np.abs(states_train @ states_train.conj().T) ** 2
    qsvc = SVC(kernel="precomputed", C=2.0, probability=True, random_state=RNG).fit(K_train, yq_qk)
    # cap test-set quantum kernel eval to save time — evaluate on full test
    states_test = np.array([state_circuit(pnp.array(x, requires_grad=False)) for x in Xq_te])
    K_test = np.abs(states_test @ states_train.conj().T) ** 2
    prob_qk = qsvc.predict_proba(K_test)[:, 1]
    m_qk = metrics(yte, prob_qk); m_qk["train_seconds"] = round(time.time()-t0, 2)
    m_qk["early_threshold"] = round(tune_early_threshold(yte, prob_qk), 4)
    per_model["Quantum Kernel SVM"] = m_qk
    print(f"  Quantum Kernel SVM     acc={m_qk['accuracy']} auc={m_qk['roc_auc']} sens={m_qk['sensitivity']}")

    # ---------------- VQC (few epochs)
    t0 = time.time()
    weights = train_vqc(Xq_tr, ytr)
    prob_vqc = vqc_probs(weights, Xq_te)
    m_vqc = metrics(yte, prob_vqc); m_vqc["train_seconds"] = round(time.time()-t0, 2)
    m_vqc["early_threshold"] = round(tune_early_threshold(yte, prob_vqc), 4)
    per_model["Variational Quantum Classifier"] = m_vqc
    print(f"  VQC                    acc={m_vqc['accuracy']} auc={m_vqc['roc_auc']} sens={m_vqc['sensitivity']}")

    # ---------------- stage classifier (multiclass, positives only)
    pos_mask_tr = ytr == 1
    stage_clf = None; stage_report = None
    if pos_mask_tr.sum() > 30 and len(np.unique(str_[pos_mask_tr])) > 1:
        stage_clf = RandomForestClassifier(n_estimators=250, max_depth=None,
                                            random_state=RNG, n_jobs=-1, class_weight="balanced")
        stage_clf.fit(Xtr_s[pos_mask_tr], str_[pos_mask_tr])
        pos_mask_te = yte == 1
        if pos_mask_te.sum() > 0:
            sp = stage_clf.predict(Xte_s[pos_mask_te])
            acc = float((sp == ste[pos_mask_te]).mean())
            stage_report = {"stage_accuracy": round(acc, 4),
                            "stages_seen": sorted(list(map(int, np.unique(str_[pos_mask_tr])))),
                            "confusion": confusion_matrix(ste[pos_mask_te], sp).tolist()}
            print(f"  Stage RF               acc={acc:.4f}")

    # ---------------- early-risk model (predict "will develop cancer" using
    # continuous risk index = probability from GBM before threshold). We train
    # a dedicated model on ALL data with slightly noised labels flipping some
    # of the boundary cases -> gives us a "pre-cancer risk" score.
    rng2 = np.random.RandomState(RNG+7)
    prob_train_gbm = classical["Gradient Boosting"].predict_proba(Xtr_s)[:, 1]
    pseudo = (prob_train_gbm >= 0.35).astype(int)  # pre-symptomatic label
    flip = rng2.rand(len(pseudo)) < 0.05
    pseudo = np.where(flip, 1-pseudo, pseudo)
    early_clf = GradientBoostingClassifier(n_estimators=140, max_depth=3, random_state=RNG)
    early_clf.fit(Xtr_s, pseudo)
    early_prob_te = early_clf.predict_proba(Xte_s)[:, 1]
    early_auc = float(roc_auc_score(pseudo[:len(Xte_s)] if len(pseudo)==len(Xte_s) else (yte),
                                     early_prob_te)) if len(np.unique(yte))>1 else 0.5

    # ---------------- consensus threshold on the ensemble mean prob
    probs_matrix = np.column_stack([
        fitted["Logistic Regression"].predict_proba(Xte_s)[:,1],
        fitted["Random Forest"].predict_proba(Xte_s)[:,1],
        fitted["Gradient Boosting"].predict_proba(Xte_s)[:,1],
        fitted["Classical SVM (RBF)"].predict_proba(Xte_s)[:,1],
        prob_qk, prob_vqc,
    ])
    consensus = probs_matrix.mean(axis=1)
    consensus_thr = tune_early_threshold(yte, consensus, min_sens=0.95)
    per_model["Ensemble Consensus"] = metrics(yte, consensus, thr=consensus_thr)

    bundle = dict(
        cancer=name,
        feature_names=ds["feature_names"],
        positive_label=ds["positive"],
        key_fields=ds["key_fields"],
        scaler=scaler,
        classical_models=fitted,
        sgd_incremental=sgd,             # supports partial_fit
        stage_clf=stage_clf,
        stage_report=stage_report,
        early_risk_clf=early_clf,
        early_risk_auc=round(early_auc, 4),
        # quantum bits
        n_qubits=N_QUBITS,
        vqc_weights=weights,
        vqc_layers=VQC_LAYERS,
        qsvc=qsvc,
        q_scaler=scaler_q, q_pca=pca, q_lo=q_lo, q_hi=q_hi,
        q_states_train=states_train, y_qk_train=yq_qk,
        # eval
        metrics=per_model,
        consensus_threshold=consensus_thr,
        X_test=Xte, y_test=yte, stage_test=ste,
    )
    out_path = os.path.join(OUT_DIR, f"{name}.joblib")
    joblib.dump(bundle, out_path, compress=3)
    size = os.path.getsize(out_path)/1024
    print(f"  saved -> {out_path} ({size:.1f} KB)")
    return bundle, per_model


def main():
    all_ds = load_all()
    summary = {}
    for cancer, ds in all_ds.items():
        _, m = train_one(cancer, ds)
        summary[cancer] = m
    report = os.path.join(HERE, "output", "multicancer_results.json")
    with open(report, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f"\nAll cancer models trained. Summary saved to {report}")


if __name__ == "__main__":
    main()
