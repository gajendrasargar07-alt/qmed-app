"""
Per-cancer scoring: four classical models, a quantum-kernel SVM and a
variational quantum classifier, blended into one score, plus a stage estimate
and an early-risk estimate. The trained bundles in models/multicancer are only
ever read here, never written.
"""
import threading

import joblib
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from . import config

WEIGHTS = {
    "Logistic Regression": 0.15,
    "Random Forest": 0.20,
    "Gradient Boosting": 0.20,
    "Classical SVM (RBF)": 0.15,
    "Quantum Kernel SVM": 0.20,
    "Variational Quantum Classifier": 0.10,
}

_cache = {}
_load_lock = threading.Lock()
_run_lock = threading.Lock()  # the quantum circuits share simulator state


def load_bundle(cancer: str) -> dict:
    with _load_lock:
        if cancer in _cache:
            return _cache[cancer]
        b = joblib.load(config.CANCER_MODELS_DIR / f"{cancer}.joblib")
        n = int(b["n_qubits"])
        dev = qml.device("default.qubit", wires=n)

        def embed(x):
            qml.AngleEmbedding(x, wires=range(n), rotation="Y")
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n - 1, 0])

        @qml.qnode(dev)
        def state_circuit(x):
            embed(x)
            return qml.state()

        @qml.qnode(dev, interface="autograd")
        def vqc_circuit(weights, x):
            embed(x)
            qml.StronglyEntanglingLayers(weights, wires=range(n))
            return qml.expval(qml.PauliZ(0))

        b["_state_circuit"] = state_circuit
        b["_vqc_circuit"] = vqc_circuit
        _cache[cancer] = b
        return b


def _quantum_input(b, X):
    Xs = b["q_scaler"].transform(X)
    Z = b["q_pca"].transform(Xs)
    lo, hi = b["q_lo"], b["q_hi"]
    Zn = (Z - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return np.clip(Zn, 0, 1) * np.pi


def band_for(score: float):
    if score >= config.HIGH_THRESHOLD:
        return "HIGH RISK", "Immediate clinical follow-up recommended"
    if score >= config.MODERATE_THRESHOLD:
        return "MODERATE RISK", "Additional screening / repeat labs advised"
    return "LOW RISK", "Routine monitoring"


def score(cancer: str, features) -> dict:
    b = load_bundle(cancer)
    x = np.atleast_2d(np.asarray(features, dtype=np.float32))
    xs = b["scaler"].transform(x)

    with _run_lock:
        per = {name: float(clf.predict_proba(xs)[0, 1])
               for name, clf in b["classical_models"].items()}

        xq = _quantum_input(b, x)
        state = b["_state_circuit"](pnp.array(xq[0], requires_grad=False))
        kernel_row = np.abs(b["q_states_train"] @ state.conj()) ** 2
        per["Quantum Kernel SVM"] = float(b["qsvc"].predict_proba(kernel_row.reshape(1, -1))[0, 1])

        w = pnp.array(b["vqc_weights"], requires_grad=False)
        v = float(b["_vqc_circuit"](w, pnp.array(xq[0], requires_grad=False)))
        per["Variational Quantum Classifier"] = (v + 1.0) / 2.0

    stage, stage_conf = None, None
    if b.get("stage_clf") is not None:
        stage = int(b["stage_clf"].predict(xs)[0])
        try:
            stage_conf = float(np.max(b["stage_clf"].predict_proba(xs)[0]))
        except Exception:
            pass

    early = float(b["early_risk_clf"].predict_proba(xs)[0, 1])

    total = sum(WEIGHTS.get(k, 0) for k in per) or 1.0
    final = sum(per[k] * WEIGHTS.get(k, 0) for k in per) / total
    band, action = band_for(final)

    return {
        "cancer": cancer,
        "per_model": {k: round(v, 4) for k, v in per.items()},
        "final_score": round(float(final), 4),
        "risk_band": band,
        "recommended_action": action,
        "predicted_stage": stage,
        "stage_confidence": None if stage_conf is None else round(stage_conf, 4),
        "early_risk_score": round(early, 4),
        "positive_label": b["positive_label"],
    }


def score_from_fields(cancer: str, fields: dict) -> dict:
    """
    Builds the feature vector from extracted values. Anything the report did
    not contain is filled with that feature's training-set mean (neutral after
    scaling) and listed under 'imputed'.
    """
    b = load_bundle(cancer)
    names = list(b["feature_names"])
    means = b["scaler"].mean_
    vec, imputed = [], []
    for i, name in enumerate(names):
        if name in fields:
            vec.append(float(fields[name]))
        else:
            vec.append(float(means[i]))
            imputed.append(name)
    out = score(cancer, vec)
    out["imputed"] = imputed
    return out


def warm_up():
    for c in config.CANCERS:
        load_bundle(c)
