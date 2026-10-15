# Speech Exercise Tracker (Phase 1)

Mobile-friendly web app where a clinician assigns speech exercises to patients, patients
record them on their phone, and the clinician listens back to a noise-reduced,
volume-levelled copy (or the untouched original).

Exercises: sustained "ahh", repeated "pa-ta-ka", and reading a fixed sentence.

## Run

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
$env:DOCTOR_SIGNUP_CODE = "choose-a-secret"   # required to register as a doctor
python app.py                                  # http://localhost:5001
```

Tests: `python -m unittest tests.test_flow -v`

## Using it

1. A doctor registers with the sign-up code and gets a 6-character **invite code** on their home page.
2. Patients register with that invite code and are linked to the doctor.
3. The doctor opens a patient, assigns exercises and a weekly target.
4. The patient taps an exercise, records, listens back and submits.
5. The doctor opens the recording and toggles **Cleaned / Original**, with 0.75× and 0.5× playback.

## Nearby clinicians

Patients who are not yet linked to a clinician see a "Find a specialist near you" list (and can
enter an invite code later). The same widget appears in the screener after a positive result.
It sorts the directory in `../shared/doctors_chennai.json` by straight-line distance, using the
browser location or a Chennai area picked from a list. Coordinates are sent by POST, used for that
request only, and never stored.

**The shipped directory is demo data** (`"demo": true`, no phone numbers). Replace it with
hand-verified entries (real phone number, `verified_on` date, `demo` removed) before real use.
Shared tests: `python ../shared/test_nearby.py`.

## Live clinician search (Google Maps)

Both apps offer "Find a specialist near you" to everyone, whatever the screening result. With a Google key it
searches Google Maps live for nearby speech therapists, neurologists, ENT and GPs, with Call, WhatsApp (mobile
numbers only), Website and Directions buttons. Without a key, or if Google fails, it uses the saved list.

1. In Google Cloud: enable **Places API (New)** and billing; restrict the key to that API only; set a daily
   quota/budget alert.
2. Copy `../.env.example` to `../.env` and put the key in the copy. Never share the file or paste the key anywhere.
3. Check it: `python ../shared/places.py --check "13.0418,80.2341"` (prints results or the exact reason it failed).
4. Restart the app.

The key stays on the server and is never sent to the browser. Searches are capped per user/IP (30 an hour) to
protect your quota, and behind a hosting proxy set `TRUSTED_PROXIES=1` so the cap is per visitor.
Google's terms limit storing Places content, so results are only cached in memory for a few minutes.

## Testing on a phone

Browsers only allow microphone access on **HTTPS or localhost**, so opening
`http://<your-pc-ip>:5001` on a phone will not record (the WAV-upload fallback still works).
For phone testing, expose the app over HTTPS with a tunnel, e.g. `cloudflared tunnel --url http://localhost:5001`
or `ngrok http 5001`, and set `COOKIE_SECURE=1`.

## How the audio is handled

- The phone records with the browser's own noise suppression / auto-gain **turned off** and
  uploads 16 kHz mono WAV.
- The server keeps the original untouched and writes a cleaned copy: 80 Hz high-pass,
  spectral-gating noise reduction, silence trimming, loudness normalisation to -16 LUFS,
  peak limiting.
- The cleaned copy is for **listening only**. When Phase 2 adds scoring, features must be
  computed from the original, because denoising changes jitter, shimmer, HNR and MFCCs.

## Configuration

| Variable | Purpose |
|---|---|
| `DOCTOR_SIGNUP_CODE` | Code required to create a doctor account (default `chennai-pilot` — change it) |
| `SECRET_KEY` | Session signing key (otherwise generated once into `instance/secret_key`) |
| `COOKIE_SECURE=1` | Send the session cookie over HTTPS only (set this in any HTTPS deployment) |

Data lives in `instance/` (SQLite database and recordings). Back it up and keep it out of git.

## Known limits of this prototype

- Single-process SQLite and synchronous audio processing: fine for a pilot, not for scale.
- No login rate-limiting, password reset or email verification yet.
- No consent screen, retention policy or audit log yet. These are required before real
  patient data is collected (see Phase 4 of the plan).
- The reading sentence is English. For Tamil-speaking patients, add a clinician-approved
  Tamil sentence to `EXERCISES` in `db.py`.
