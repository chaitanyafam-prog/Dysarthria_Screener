"""Step 2 (Kaggle GPU, internet ON): one pass that extracts, per audio file,
   * the same 85 handcrafted features the deployed app uses, and
   * layer-wise mean+std pooled embeddings from WavLM / wav2vec2 / HuBERT.

Because both come from the same loop, rows are aligned by construction and every failure is logged
(the old notebook silently dropped 9 TORGO files, so its rows could not be matched to filenames).

Output: /kaggle/working/features/chunk_00000.npz, chunk_00001.npz, ...  (resumable: finished chunks are skipped)

Usage in a Kaggle notebook (Settings: Accelerator = GPU, Internet = On):
    !pip install -q praat-parselmouth transformers
    import sys; sys.path.insert(0, "/kaggle/input/<your-dataset-name>")
    exec(open("/kaggle/input/<your-dataset-name>/step2_extract.py").read())
"""
import csv
import glob
import os
import re
import time
import warnings

import librosa
import numpy as np
import parselmouth
import torch
from parselmouth.praat import call
from transformers import AutoFeatureExtractor, AutoModel

warnings.filterwarnings("ignore")

# ----------------------------- configuration -----------------------------
TORGO_CSV = "/kaggle/input/datasets/iamhungundji/dysarthria-detection/torgo_data/data.csv"
TORGO_BASE = "/kaggle/input/datasets/iamhungundji/dysarthria-detection/"
UA_ROOT = "/kaggle/input/datasets/aryashah2k/noise-reduced-uaspeech-dysarthria-dataset/"
UA_DYS_SPEAKERS = ["F02", "F03", "F04", "F05", "M01", "M04", "M05", "M07", "M08", "M09",
                   "M10", "M11", "M12", "M14", "M16"]
UA_CTL_SPEAKERS = ["CF02", "CF03", "CF04", "CF05", "CM01", "CM04", "CM05", "CM06", "CM08",
                   "CM09", "CM10", "CM12", "CM13"]
UA_PER_SPEAKER = 100          # raise to 200-300 later for more within-speaker variety (costs time)
MODELS = {
    "wavlm": "microsoft/wavlm-base-plus",
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
}
SR = 16000
MAX_SECONDS = 10              # longer clips are truncated to bound GPU memory
CHUNK = 500
OUT_DIR = "/kaggle/working/features"
SEED = 42


# ----------------------------- file list -----------------------------
def build_manifest():
    rows = []
    with open(TORGO_CSV, newline="") as f:
        for r in csv.DictReader(f):
            path = os.path.join(TORGO_BASE, r["filename"])
            m = re.match(r"([A-Z]+\d+)", os.path.basename(path))
            rows.append(("torgo", "T_" + (m.group(1) if m else "unknown"),
                         1 if r["is_dysarthria"] == "dysarthria" else 0, path))

    rng = np.random.default_rng(SEED)
    for speakers, folder, label in [(UA_DYS_SPEAKERS, "noisereduced-uaspeech", 1),
                                    (UA_CTL_SPEAKERS, "noisereduced-uaspeech-control", 0)]:
        for spk in speakers:
            files = sorted(glob.glob(os.path.join(UA_ROOT, folder, spk, "*.wav")))
            take = rng.choice(len(files), size=min(UA_PER_SPEAKER, len(files)), replace=False)
            rows += [("uaspeech", "U_" + spk, label, files[i]) for i in sorted(take)]
    return rows


# ----------------------------- handcrafted features (identical to the deployed app) -----------------------------
def extract_rich_features(path, sr=SR):
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


# ----------------------------- embeddings -----------------------------
def pool_hidden_states(hidden_states):
    """tuple of (1, T, H) tensors -> float16 array (n_layers, 2H): mean and std over time per layer."""
    h = torch.stack([s[0] for s in hidden_states])           # (L, T, H)
    pooled = torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False)], dim=-1)
    return pooled.float().cpu().numpy().astype(np.float16)


def load_models(device):
    loaded = {}
    for key, name in MODELS.items():
        fe = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).to(device).eval()
        loaded[key] = (fe, model)
        print("loaded", name)
    return loaded


@torch.inference_mode()
def embed(models, audio, device):
    out = {}
    for key, (fe, model) in models.items():
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt").to(device)
        hs = model(**inputs, output_hidden_states=True).hidden_states
        out[key] = pool_hidden_states(hs)
    return out


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, duration=MAX_SECONDS)
    if len(y) < SR // 2:
        y = np.pad(y, (0, SR // 2 - len(y)))                  # models need a minimum length
    return y


# ----------------------------- main loop -----------------------------
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "(use a GPU runtime; CPU will take hours)" if device == "cpu" else "")
    manifest = build_manifest()
    print(f"{len(manifest)} files, {len(set(m[1] for m in manifest))} speakers")

    models = load_models(device)
    failures = []
    t0 = time.time()
    for start in range(0, len(manifest), CHUNK):
        out_path = os.path.join(OUT_DIR, f"chunk_{start // CHUNK:05d}.npz")
        if os.path.exists(out_path):
            continue
        rows, hand, embs = [], [], {k: [] for k in MODELS}
        for corpus, speaker, label, path in manifest[start:start + CHUNK]:
            try:
                y = load_audio(path)
                e = embed(models, y, device)
                h = extract_rich_features(path)
            except Exception as exc:
                failures.append((path, repr(exc)))
                continue
            rows.append((corpus, speaker, label, path))
            hand.append(h)
            for k in MODELS:
                embs[k].append(e[k])
        np.savez_compressed(
            out_path,
            corpus=np.array([r[0] for r in rows]), speaker=np.array([r[1] for r in rows]),
            label=np.array([r[2] for r in rows]), path=np.array([r[3] for r in rows]),
            hand=np.array(hand, dtype=np.float64),
            **{f"emb_{k}": np.array(v) for k, v in embs.items()})
        done = min(start + CHUNK, len(manifest))
        print(f"{done}/{len(manifest)} done, {len(failures)} failures, {(time.time() - t0) / 60:.1f} min")

    if failures:
        with open(os.path.join(OUT_DIR, "failures.csv"), "w", newline="") as f:
            csv.writer(f).writerows([("path", "error"), *failures])
        print(f"{len(failures)} files failed; see failures.csv. First: {failures[0]}")
    print("finished. Save the notebook output (or the features/ folder as a dataset) before step 3.")


if __name__ == "__main__":   # also true when exec()'d inside a notebook cell
    run()
