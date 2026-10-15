"""Rebuild the six feature files that step 1 needs:
     X_rich.npy  y_rich.npy  groups.npy        (TORGO)
     X_ua.npy    y_ua.npy    groups_ua.npy     (UASpeech)

Paste into ONE Kaggle notebook cell (with the TORGO and UASpeech datasets attached) and run it.
Uses 4 worker processes (~5-8 minutes in total), skips any dataset whose three files already exist,
and writes failures_*.csv instead of silently dropping files.
"""
import csv
import glob
import os
import re
import warnings

N_JOBS = globals().get("N_JOBS", 4)
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)       # Kaggle allows ~4 cores but reports the host's count

import librosa
import numpy as np
import parselmouth
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ----------------------------- configuration -----------------------------
OUT = globals().get("OUT", "/kaggle/working/")
TORGO_CSV = globals().get("TORGO_CSV", "/kaggle/input/datasets/iamhungundji/dysarthria-detection/torgo_data/data.csv")
TORGO_BASE = globals().get("TORGO_BASE", "/kaggle/input/datasets/iamhungundji/dysarthria-detection/")
UA_ROOT = globals().get("UA_ROOT", "/kaggle/input/datasets/aryashah2k/noise-reduced-uaspeech-dysarthria-dataset/")
UA_PER_SPEAKER = globals().get("UA_PER_SPEAKER", 100)
FORCE = globals().get("FORCE", False)                # True = rebuild even if the files exist
SEED = 42
UA_DYS = globals().get("UA_DYS", ["F02", "F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09",
                                  "M10", "M11", "M12", "M14", "M16"])
UA_CTL = globals().get("UA_CTL", ["CF02", "CF03", "CF04", "CF05", "CM01", "CM04", "CM05", "CM06", "CM08",
                                  "CM09", "CM10", "CM12", "CM13"])


# ----------------------------- the 85 features (same as the deployed app) -----------------------------
def extract_rich_features(path, sr=16000):
    # Imported here, not at module level: Praat's `call` is a native object that cannot be pickled,
    # so a global reference would break sending this function to the worker processes.
    from parselmouth.praat import call

    y, sr = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                                 delta.mean(axis=1), delta.std(axis=1),
                                 delta2.mean(axis=1), delta2.std(axis=1)])
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rmse = librosa.feature.rms(y=y).mean()

    snd = parselmouth.Sound(path)
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    try:
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        jitter = 0
    try:
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = 0
    try:
        hnr = call(call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0), "Get mean", 0, 0)
    except Exception:
        hnr = 0
    try:
        formant = call(snd, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)
        f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
    except Exception:
        f1, f2 = 0, 0
    return np.concatenate([mfcc_feats, np.array([jitter, shimmer, hnr, f1, f2, zcr, rmse])])


def safe_extract(path):
    """(features, None) on success, (None, error text) on failure: never raises, never hides a failure."""
    try:
        return extract_rich_features(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_extraction(rows, name):
    """rows = [(path, label, speaker)] -> arrays; failures are written to failures_<name>.csv."""
    paths = [r[0] for r in rows]
    job = Parallel(n_jobs=N_JOBS, batch_size=8, return_as="generator")(delayed(safe_extract)(p) for p in paths)
    results = list(tqdm(job, total=len(paths), desc=name))

    keep, failures = [], []
    for row, (feat, err) in zip(rows, results):
        if feat is None:
            failures.append((row[0], err))
        else:
            keep.append((feat, row[1], row[2]))
    if failures:
        with open(os.path.join(OUT, f"failures_{name}.csv"), "w", newline="") as f:
            csv.writer(f).writerows([("path", "error"), *failures])
    print(f"{name}: {len(keep)} ok, {len(failures)} failed"
          f"{' (see failures_' + name + '.csv; clips under ~0.3 s cannot give delta features)' if failures else ''}")
    X = np.array([k[0] for k in keep])
    y = np.array([k[1] for k in keep])
    g = np.array([k[2] for k in keep])
    return X, y, g


def already_done(*names):
    return (not FORCE) and all(os.path.exists(os.path.join(OUT, n)) for n in names)


# ----------------------------- TORGO -----------------------------
if already_done("X_rich.npy", "y_rich.npy", "groups.npy"):
    print("TORGO files already exist: skipping (set FORCE = True to rebuild)")
else:
    rows = []
    with open(TORGO_CSV, newline="") as f:
        for r in csv.DictReader(f):
            path = os.path.join(TORGO_BASE, r["filename"])
            m = re.match(r"([A-Z]+\d+)", os.path.basename(path))
            rows.append((path, 1 if r["is_dysarthria"] == "dysarthria" else 0, m.group(1) if m else "unknown"))
    X, y, g = run_extraction(rows, "torgo")
    np.save(os.path.join(OUT, "X_rich.npy"), X)
    np.save(os.path.join(OUT, "y_rich.npy"), y)
    np.save(os.path.join(OUT, "groups.npy"), g)
    print("saved X_rich.npy, y_rich.npy, groups.npy", X.shape)

# ----------------------------- UASpeech -----------------------------
if already_done("X_ua.npy", "y_ua.npy", "groups_ua.npy"):
    print("UASpeech files already exist: skipping (set FORCE = True to rebuild)")
else:
    rng = np.random.default_rng(SEED)
    rows = []
    for speakers, folder, label in [(UA_DYS, "noisereduced-uaspeech", 1),
                                    (UA_CTL, "noisereduced-uaspeech-control", 0)]:
        for spk in speakers:
            files = sorted(glob.glob(os.path.join(UA_ROOT, folder, spk, "*.wav")))
            pick = rng.choice(len(files), size=min(UA_PER_SPEAKER, len(files)), replace=False) if files else []
            rows += [(files[i], label, spk) for i in sorted(pick)]
    print(f"UASpeech: {len(rows)} files selected from {len(UA_DYS) + len(UA_CTL)} speakers")
    X_ua, y_ua, groups_ua = run_extraction(rows, "ua")
    np.save(os.path.join(OUT, "X_ua.npy"), X_ua)
    np.save(os.path.join(OUT, "y_ua.npy"), y_ua)
    np.save(os.path.join(OUT, "groups_ua.npy"), groups_ua)
    print("saved X_ua.npy, y_ua.npy, groups_ua.npy", X_ua.shape)

print("\nDone. Now run step 1.")
