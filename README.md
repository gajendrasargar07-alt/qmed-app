# QMed

QMed turns a lab or pathology report into a plain-language read. Upload a PDF, a photo or a text file; the app reads it, works out which type of report it is, scores the values with a set of classical and quantum models, and explains the result.

It supports breast, lung, prostate, colon and cervical reports.

## Run it

You need Python 3.10 or newer, plus Tesseract and Poppler for scanned reports and photos.

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr poppler-utils
# macOS
brew install tesseract poppler

pip install -r requirements.txt
python app.py
```

Open http://localhost:8000. Set `PORT` to use a different port. `GET /api/health` reports whether OCR is available.

### Docker

```bash
docker build -t qmed .
docker run -p 8000:8000 qmed
```

## How it works

```
upload -> text (PDF text layer, or OCR for scans and photos)
       -> check it is a medical report
       -> pick the report type (keyword hints, then a TF-IDF router)
       -> read the lab values
       -> score: 4 classical models + quantum kernel SVM + variational quantum classifier
       -> combine into one score, plus stage and early-risk estimates
       -> plain-language result
```

If a report is missing any of the values the models rely on for its type, the app lists what is missing instead of giving a result. Other missing values are filled with the training average and the result is marked as less certain. The switch is `REQUIRE_KEY_FIELDS` in `qmed/config.py`.

## Project layout

| Path | What it is |
|---|---|
| `app.py` | Flask server: the page, `POST /api/analyze`, `GET /api/health` |
| `qmed/` | Backend: OCR and value extraction, validator, scoring, pipeline, result wording |
| `web/` | The interface (one page, no build step) |
| `models/` | Trained models. Read-only at runtime. |
| `models.sha256` | Checksums of everything in `models/`; a test checks them |
| `training/` | Scripts that train the models |
| `tests/` | Backend tests and sample reports |

## Tests

```bash
python -m unittest discover -s tests -v
```

## Retraining

```bash
python training/train_all.py      # per-cancer models
python training/train_router.py   # report-type router
```

Both write to `training/output/`, never to `models/`. Compare the results with `training/reference/multicancer_results.json`, then copy files across yourself if you want to replace the shipped models (and regenerate `models.sha256`).

## About the data

The breast model is trained on the public Breast Cancer Wisconsin dataset. The lung, prostate, colon and cervical models are trained on simulated cohorts built from published clinical ranges, not on real patients. The app says this in its About sheet. Before offering those four types to the public, validate them on real, consented clinical data and get the regulatory and legal review your market requires.

## Privacy

Uploaded reports are processed in memory and are not written to disk or logged. Recent results are kept in the browser's local storage on the person's own device and can be cleared from the home screen.
