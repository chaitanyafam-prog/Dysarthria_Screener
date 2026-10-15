"""Leak-free, speaker-independent evaluation helpers.

Rules this module enforces:
* every model passed in is a scikit-learn Pipeline, so imputing/scaling is fit inside each fold;
* folds are whole speakers (leave-one-speaker-out);
* results are reported per speaker, with a speaker-level bootstrap confidence interval,
  because with ~40 speakers the pooled accuracy alone hides how unstable the estimate is.
"""
import os

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict


# Kaggle containers report the host's core count but only allow ~4; n_jobs=-1 can oversubscribe and crawl.
N_JOBS = min(4, os.cpu_count() or 1)


def loso_scores(model, X, y, groups, n_jobs=N_JOBS):
    """Leave-one-speaker-out decision scores (> 0 means dysarthria)."""
    return cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut(),
                             n_jobs=n_jobs, method="decision_function")


def summarize(y, scores, groups, corpus=None, n_boot=2000, seed=0):
    """Pooled and per-speaker metrics plus a 95% speaker-bootstrap CI for accuracy."""
    y, scores, groups = np.asarray(y), np.asarray(scores), np.asarray(groups)
    pred = (scores > 0).astype(int)
    correct = pred == y

    speakers = np.unique(groups)
    n = np.array([(groups == s).sum() for s in speakers])
    c = np.array([correct[groups == s].sum() for s in speakers])
    per_speaker = c / n

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(speakers), size=(n_boot, len(speakers)))
    boot = c[idx].sum(axis=1) / n[idx].sum(axis=1)

    out = {
        "n_speakers": len(speakers),
        "acc": float(correct.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "spk_mean": float(per_speaker.mean()),
        "spk_std": float(per_speaker.std()),
        "spk_min": float(per_speaker.min()),
        "per_speaker": {str(s): round(float(a), 3) for s, a in zip(speakers, per_speaker)},
    }
    if corpus is not None:
        corpus = np.asarray(corpus)
        for name in np.unique(corpus):
            out[f"acc_{name}"] = float(correct[corpus == name].mean())
    return out


def cross_corpus(model, X, y, corpus, train_on, test_on):
    """Fit on one corpus, test on the other. A collapse means the model learned the corpus."""
    X, y, corpus = np.asarray(X), np.asarray(y), np.asarray(corpus)
    tr, te = corpus == train_on, corpus == test_on
    m = clone(model).fit(X[tr], y[tr])
    scores = m.decision_function(X[te])
    pred = (scores > 0).astype(int)
    return {
        "acc": float((pred == y[te]).mean()),
        "bal_acc": float(balanced_accuracy_score(y[te], pred)),
        "auc": float(roc_auc_score(y[te], scores)),
    }


def pick_holdout(groups, y, corpus, frac=0.2, seed=0):
    """Reserve ~frac of speakers (stratified by corpus and label) as a final test set.

    Choose the model/layer using only the remaining speakers, then score the holdout once.
    Selecting among many layers on the same folds you report is optimistic; this is the cure.
    """
    groups, y, corpus = np.asarray(groups), np.asarray(y), np.asarray(corpus)
    rng = np.random.default_rng(seed)
    strata = {}
    for s in np.unique(groups):
        m = groups == s
        strata.setdefault((corpus[m][0], int(round(y[m].mean()))), []).append(s)
    held = []
    for key in sorted(strata):
        members = list(strata[key])
        rng.shuffle(members)
        held += members[:max(1, round(frac * len(members)))]
    return sorted(held)
