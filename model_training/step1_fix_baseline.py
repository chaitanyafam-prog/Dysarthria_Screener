"""Step 1 (CPU, ~2 minutes): re-score your existing 85-feature SVM honestly.

Run in your existing Kaggle notebook (the one that already has /kaggle/working/X_rich.npy etc.),
after uploading this folder as a dataset:

    import sys; sys.path.insert(0, "/kaggle/input/<your-dataset-name>")
    exec(open("/kaggle/input/<your-dataset-name>/step1_fix_baseline.py").read())

It shows how much of the reported 75% survives two fixes:
  (a) speaker IDs that collide between corpora (F03, F04, M01, M04, M05 exist in both),
  (b) imputer/scaler fitted on all data before cross-validation.
"""
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evalkit import N_JOBS, cross_corpus, loso_scores, summarize

W = globals().get("WORK_DIR", "/kaggle/working/")
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


tick(f"workers: {N_JOBS}")
X_t, y_t, g_t = np.load(W + "X_rich.npy"), np.load(W + "y_rich.npy"), np.load(W + "groups.npy", allow_pickle=True)
X_u, y_u, g_u = np.load(W + "X_ua.npy"), np.load(W + "y_ua.npy"), np.load(W + "groups_ua.npy", allow_pickle=True)

X = np.concatenate([X_t, X_u])
y = np.concatenate([y_t, y_u])
corpus = np.array(["torgo"] * len(y_t) + ["uaspeech"] * len(y_u))
g_old = np.concatenate([g_t, g_u]).astype(str)
g_new = np.array([("T_" if c == "torgo" else "U_") + s for c, s in zip(corpus, g_old)])

print(f"samples {len(y)} | speakers by old IDs: {len(np.unique(g_old))} | by corpus-prefixed IDs: {len(np.unique(g_new))}")
print("colliding IDs:", sorted(set(g_t.astype(str)) & set(g_u.astype(str))))


def svm():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))


tick("starting (a): about 1-3 minutes per stage")
# (a) your original protocol: preprocessing fitted on ALL data, merged speaker IDs.
Xi = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))
pred = cross_val_predict(SVC(kernel="rbf", C=1.0, gamma="scale"), Xi, y, groups=g_old, cv=LeaveOneGroupOut(), n_jobs=N_JOBS)
tick(f"(a) original protocol                       accuracy {np.mean(pred == y):.3f}")

# (b) corpus-prefixed IDs, still globally preprocessed
pred = cross_val_predict(SVC(kernel="rbf", C=1.0, gamma="scale"), Xi, y, groups=g_new, cv=LeaveOneGroupOut(), n_jobs=N_JOBS)
tick(f"(b) + correct speaker IDs                   accuracy {np.mean(pred == y):.3f}")

# (c) fully leak-free: preprocessing fitted inside each fold
tick("starting (c)")
scores = loso_scores(svm(), X, y, g_new)
r = summarize(y, scores, g_new, corpus)
print(f"(c) + preprocessing inside each fold        accuracy {r['acc']:.3f}  (95% CI {r['ci_lo']:.3f}-{r['ci_hi']:.3f})")
print(f"    balanced acc {r['bal_acc']:.3f} | AUC {r['auc']:.3f} | per-speaker mean {r['spk_mean']:.3f} +/- {r['spk_std']:.3f}, worst {r['spk_min']:.3f}")
print(f"    TORGO {r['acc_torgo']:.3f} | UASpeech {r['acc_uaspeech']:.3f}")

worst = sorted(r["per_speaker"].items(), key=lambda kv: kv[1])[:8]
print("    hardest speakers:", ", ".join(f"{s}={a:.2f}" for s, a in worst))

# Corpus-confound check: if either number is near 0.5, the model is learning recording conditions.
for a, b in [("torgo", "uaspeech"), ("uaspeech", "torgo")]:
    cc = cross_corpus(svm(), X, y, corpus, a, b)
    tick(f"cross-corpus train {a:9s} -> test {b:9s}: accuracy {cc['acc']:.3f}, AUC {cc['auc']:.3f}")
