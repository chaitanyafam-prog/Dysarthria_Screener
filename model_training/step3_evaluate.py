"""Step 3 (CPU is fine): compare the handcrafted baseline with pretrained embeddings, honestly.

Protocol
  1. Reserve ~20% of speakers (stratified by corpus and label) as a HOLDOUT that is never used to choose anything.
  2. On the remaining DEV speakers, run leave-one-speaker-out for the baseline and for every
     (model, layer) of mean+std embeddings, plus embeddings + handcrafted features.
  3. Take the best dev configuration and score it ONCE on the holdout speakers, next to the baseline.
  4. Cross-corpus check (train TORGO -> test UASpeech and back) for the winner.
  5. Decision gate: adopt only if (a) the paired per-speaker accuracy gain over the baseline on dev
     speakers has a 95% CI lower bound above 0, (b) the holdout is not worse, and (c) cross-corpus
     is not worse. If it passes, export a bundle fit on all speakers.

Usage:
    import sys; sys.path.insert(0, "/kaggle/input/<your-dataset-name>")
    FEATURES_DIR = "/kaggle/working/features"      # or the dataset where you saved step 2's output
    exec(open("/kaggle/input/<your-dataset-name>/step3_evaluate.py").read())
"""
import csv
import glob
import os

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evalkit import cross_corpus, loso_scores, pick_holdout, summarize

MODEL_IDS = {"wavlm": "microsoft/wavlm-base-plus", "wav2vec2": "facebook/wav2vec2-base",
             "hubert": "facebook/hubert-base-ls960"}
LR_C = 0.05


def load_features(features_dir):
    parts = [np.load(p, allow_pickle=False) for p in sorted(glob.glob(os.path.join(features_dir, "chunk_*.npz")))]
    if not parts:
        raise FileNotFoundError(f"no chunk_*.npz in {features_dir}")
    cat = lambda key: np.concatenate([p[key] for p in parts])
    data = {k: cat(k) for k in ("corpus", "speaker", "label", "path", "hand")}
    data["emb"] = {k[4:]: cat(k) for k in parts[0].files if k.startswith("emb_")}
    return data


def baseline_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))


def embedding_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, max_iter=500))


def combined_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=LR_C, max_iter=500))


def evaluate(data, holdout_frac=0.2, seed=0, verbose=True):
    y, g, corpus = data["label"], data["speaker"], data["corpus"]
    held = pick_holdout(g, y, corpus, holdout_frac, seed)
    dev = ~np.isin(g, held)
    say = print if verbose else (lambda *a, **k: None)
    say(f"{len(y)} samples, {len(np.unique(g))} speakers | holdout {len(held)} speakers: {held}")

    def dev_run(model, X):
        return summarize(y[dev], loso_scores(model, X[dev], y[dev], g[dev]), g[dev], corpus[dev])

    # ---- baseline (handcrafted) ----
    base_dev = dev_run(baseline_model(), data["hand"])
    say(f"\nBASELINE handcrafted SVM  dev LOSO acc {base_dev['acc']:.3f} "
        f"(CI {base_dev['ci_lo']:.3f}-{base_dev['ci_hi']:.3f}), bal {base_dev['bal_acc']:.3f}, AUC {base_dev['auc']:.3f}")

    # ---- layer sweep ----
    results, dev_full = [], {}
    for name, arr in data["emb"].items():
        n_layers = arr.shape[1]
        for layer in range(n_layers):
            X = arr[:, layer, :].astype(np.float32)
            r = dev_run(embedding_model(), X)
            dev_full[(name, layer)] = r
            results.append({"model": name, "layer": layer, "features": "embedding", **{k: r[k] for k in
                            ("acc", "ci_lo", "ci_hi", "bal_acc", "auc", "spk_mean", "spk_std", "spk_min")}})
            say(f"{name:9s} layer {layer:2d}  acc {r['acc']:.3f}  bal {r['bal_acc']:.3f}  AUC {r['auc']:.3f}")

    best = max(results, key=lambda r: r["bal_acc"])
    say(f"\nBest embedding layer on dev: {best['model']} layer {best['layer']} (bal acc {best['bal_acc']:.3f})")

    # ---- embedding + handcrafted ----
    emb_best = data["emb"][best["model"]][:, best["layer"], :].astype(np.float32)
    X_comb = np.concatenate([emb_best, data["hand"]], axis=1)
    comb = dev_run(combined_model(), X_comb)
    results.append({"model": best["model"], "layer": best["layer"], "features": "embedding+handcrafted",
                    **{k: comb[k] for k in ("acc", "ci_lo", "ci_hi", "bal_acc", "auc", "spk_mean", "spk_std", "spk_min")}})
    say(f"embedding + handcrafted     acc {comb['acc']:.3f}  bal {comb['bal_acc']:.3f}  AUC {comb['auc']:.3f}")

    use_comb = comb["bal_acc"] > best["bal_acc"]
    chosen_dev = comb if use_comb else dev_full[(best["model"], best["layer"])]
    chosen_X = X_comb if use_comb else emb_best
    chosen_factory = combined_model if use_comb else embedding_model
    chosen_name = "embedding+handcrafted" if use_comb else "embedding"

    # ---- the one honest test: holdout speakers ----
    def holdout_score(model, X):
        m = model.fit(X[dev], y[dev])
        return summarize(y[~dev], m.decision_function(X[~dev]), g[~dev], corpus[~dev])

    base_ho = holdout_score(baseline_model(), data["hand"])
    new_ho = holdout_score(chosen_factory(), chosen_X)
    say(f"\nHOLDOUT ({len(held)} unseen speakers, scored once)")
    say(f"  baseline  acc {base_ho['acc']:.3f} (CI {base_ho['ci_lo']:.3f}-{base_ho['ci_hi']:.3f})  bal {base_ho['bal_acc']:.3f}  AUC {base_ho['auc']:.3f}")
    say(f"  new       acc {new_ho['acc']:.3f} (CI {new_ho['ci_lo']:.3f}-{new_ho['ci_hi']:.3f})  bal {new_ho['bal_acc']:.3f}  AUC {new_ho['auc']:.3f}")

    # ---- cross-corpus on all data for the chosen configuration ----
    cc = {}
    for a, b in [("torgo", "uaspeech"), ("uaspeech", "torgo")]:
        cc[f"{a}->{b}"] = {"baseline": cross_corpus(baseline_model(), data["hand"], y, corpus, a, b),
                           "new": cross_corpus(chosen_factory(), chosen_X, y, corpus, a, b)}
        say(f"cross-corpus {a:8s}->{b:8s} baseline acc {cc[f'{a}->{b}']['baseline']['acc']:.3f}   new acc {cc[f'{a}->{b}']['new']['acc']:.3f}")

    # ---- decision gate ----
    # Main evidence: paired per-speaker accuracy gain over the baseline on the ~35 dev speakers.
    # Because the best of many layers was picked on those same speakers, the gain is optimistic;
    # the holdout and cross-corpus checks below guard against over-selection.
    shared = sorted(set(chosen_dev["per_speaker"]) & set(base_dev["per_speaker"]))
    diff = np.array([chosen_dev["per_speaker"][s] - base_dev["per_speaker"][s] for s in shared])
    boot = np.random.default_rng(seed).choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
    gain, gain_lo = float(diff.mean()), float(np.percentile(boot, 2.5))
    holdout_ok = new_ho["bal_acc"] >= base_ho["bal_acc"]
    cross_ok = all(v["new"]["acc"] >= v["baseline"]["acc"] - 0.02 for v in cc.values())
    passed = gain_lo > 0 and holdout_ok and cross_ok
    say(f"\nDECISION GATE: dev per-speaker gain {gain:+.3f} (95% CI lower bound {gain_lo:+.3f}, need > 0); "
        f"holdout not worse: {holdout_ok}; cross-corpus not worse: {cross_ok}"
        f" -> {'ADOPT the new model' if passed else 'KEEP the current model'}")

    return {"holdout": held, "results": results, "best": best, "chosen": chosen_name,
            "baseline_holdout": base_ho, "new_holdout": new_ho, "cross_corpus": cc, "passed": passed,
            "dev_gain": gain, "dev_gain_ci_lo": gain_lo,
            "chosen_X": chosen_X, "chosen_factory": chosen_factory}


def export_bundle(data, out, path):
    """Fit the chosen configuration on ALL speakers and save everything inference needs."""
    X, y = out["chosen_X"], data["label"]
    pipe = out["chosen_factory"]().fit(X, y)
    bundle = {
        "pipeline": pipe,
        "hf_model": MODEL_IDS[out["best"]["model"]],
        "layer": int(out["best"]["layer"]),
        "pooling": "mean+std over time of hidden_states[layer]",
        "uses_handcrafted": out["chosen"] == "embedding+handcrafted",
        "sample_rate": 16000,
        "max_seconds": 10,
        "holdout_accuracy": out["new_holdout"]["acc"],
        "baseline_holdout_accuracy": out["baseline_holdout"]["acc"],
    }
    joblib.dump(bundle, path)
    return bundle


def save_results(out, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out["results"][0]))
        w.writeheader()
        w.writerows(out["results"])


if __name__ == "__main__":
    FEATURES_DIR = globals().get("FEATURES_DIR", "/kaggle/working/features")
    data = load_features(FEATURES_DIR)
    out = evaluate(data)
    save_results(out, "/kaggle/working/layer_results.csv")
    if out["passed"]:
        export_bundle(data, out, "/kaggle/working/dysarthria_bundle.joblib")
        print("saved /kaggle/working/dysarthria_bundle.joblib")
    else:
        print("gate not passed; no bundle exported (layer_results.csv still saved for inspection)")
