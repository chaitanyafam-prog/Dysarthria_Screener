# Improving the dysarthria model (Kaggle kit)

Everything here is written to run in your existing Kaggle notebook (the one with TORGO and
UASpeech attached). Nothing here changes the deployed apps until step 3 says the new model is
genuinely better.

## What the old notebook got wrong (so the 75% is optimistic)

| Issue | Where | Effect |
|---|---|---|
| Speaker IDs collide across corpora (`F03, F04, M01, M04, M05` exist in both) | cell 15 | 43 real speakers counted as 38; some different people share a fold |
| Imputer + scaler fit on all data before cross-validation | cell 15 | held-out speaker leaks into the scaling |
| Only 100 of ~5,000 files per UASpeech speaker used | cell 5 | most of the available data is unused |
| `except: pass` while extracting TORGO | cell 11 | 9 files vanished silently, so rows can't be matched to filenames |
| No test on phone-mic audio | n/a | the app will receive phone recordings, which neither corpus resembles |

## Setup (once)

1. Upload this whole `model_training/` folder to Kaggle as a **private dataset** (e.g. `dysarthria-code`).
2. In your notebook, add that dataset to the inputs.

## Step 1 (CPU, ~2 min): honest re-score of the current model

Needs your existing `/kaggle/working/X_rich.npy`, `X_ua.npy` etc.

```python
import sys; sys.path.insert(0, "/kaggle/input/dysarthria-code")
exec(open("/kaggle/input/dysarthria-code/step1_fix_baseline.py").read())
```

It prints your accuracy under the original protocol, then after fixing the speaker IDs, then fully
leak-free, with a confidence interval, the hardest speakers, and train-on-one-corpus/test-on-the-other.
**Paste the output back to me.** If cross-corpus accuracy is near 50%, the model is partly learning
the recording setup, which matters more than any model choice.

## Step 2 (GPU, Internet ON, ~30-45 min): extract features once

Notebook settings: Accelerator = GPU (T4), Internet = On (needed to download the Hugging Face models;
this requires a phone-verified Kaggle account).

```python
!pip install -q praat-parselmouth transformers
import sys; sys.path.insert(0, "/kaggle/input/dysarthria-code")
exec(open("/kaggle/input/dysarthria-code/step2_extract.py").read())
```

Per file it saves the 85 handcrafted features (identical to the deployed app) and layer-wise
mean+std embeddings from WavLM, wav2vec2 and HuBERT, in resumable chunks under
`/kaggle/working/features/`. Failures are listed in `failures.csv` instead of being skipped silently.
If the session dies, re-run the cell: finished chunks are skipped.

Then **Save Version -> Save & Run All** (or save `features/` as a Kaggle dataset) so the files persist.

## Step 3 (CPU, ~15-30 min): compare fairly and decide

```python
FEATURES_DIR = "/kaggle/working/features"
exec(open("/kaggle/input/dysarthria-code/step3_evaluate.py").read())
```

Protocol: hold out ~20% of speakers that are never used to choose anything; sweep every model x layer
with leave-one-speaker-out on the rest; score the winner once on the holdout; check cross-corpus; then
apply the gate: adopt only if the **paired per-speaker gain over the baseline has a 95% CI lower bound
above 0**, the holdout is not worse, and cross-corpus is not worse. If it passes it writes
`/kaggle/working/dysarthria_bundle.joblib`; either way it writes `layer_results.csv`.
**Paste the printed output back to me.**

## What happens after

- Gate passed: I add embedding extraction (CPU `torch` + `transformers`) to the screener app, load the
  bundle, and update the README numbers. Note the heavier install (`torch` is large) and a few seconds
  more latency per clip.
- Gate not passed: we keep the current model and instead look at the findings from step 1
  (cross-corpus collapse, hardest speakers) and at collecting more varied speakers.

## Not yet covered (important)

- **Phone-microphone validation.** TORGO/UASpeech are studio or headset recordings. Before trusting any
  number for real use, record a small labelled set of phone clips (via the tracker, with clinician
  labels and consent) and score the final model on them.
- **Tamil and Indian-English speech.** Both corpora are English with US/Canadian speakers. Accuracy on
  Chennai speakers is unknown and could be materially different.
- Tested here: the evaluation logic on synthetic data (it finds a planted signal and refuses to adopt
  a model with none) and the extraction script on real WavLM weights. Not tested: the actual Kaggle
  data run.
