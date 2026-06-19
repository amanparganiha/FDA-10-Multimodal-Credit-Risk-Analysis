# Deploying the FDA-10 Credit Rating Dashboard (Streamlit Cloud)

## What the deployed app needs (must be on GitHub)
The Streamlit app loads these files. **All of them must be committed to the repo**:

```
requirements.txt                                       (root)
dashboard/dashboard_app.py                             (main file / entry point)
dashboard/model_loader.py
dashboard/credit_ratings_multimodal_final.csv          (~18 MB)
dashboard/models/all_multiclass_gradient_boosting.pkl
dashboard/models/all_binary_gradient_boosting.pkl
dashboard/models/tfidf_vectorizer.joblib
```

The remaining files in `dashboard/models/` are optional (the app doesn't load them).

## Do NOT upload
- `project/` — a full duplicate of the repo (doubles size, causes confusion).
- `data/processed/model_artifacts/` — research model files, 40–73 MB each. GitHub
  warns above 50 MB and **rejects files above 100 MB**; these aren't used by the app.
- `__pycache__/`, `.ipynb_checkpoints/`.

A `.gitignore` is included that already excludes the above.

## Streamlit Cloud settings
- **Repository:** amanparganiha/FDA-10-Multimodal-Credit-Risk-Analysis
- **Branch:** main
- **Main file path:** `dashboard/dashboard_app.py`   ← verify this in the app's Settings
- **Python version: 3.13**  ← REQUIRED. Set this under "Advanced settings" when deploying.

### Why Python 3.13 (not 3.14)
scikit-learn 1.7.1 (the version the models were pickled with) has no prebuilt wheel for
Python 3.14, so 3.14 tries to build it (and scipy) from source and fails with
`ERROR: Unknown compiler(s): gfortran`. Python 3.13 installs everything from wheels.
If you ever see a `gfortran` / `Failed to build scikit-learn` error in the build log,
the Python version is the cause — set it to 3.13 and reboot/redeploy.

## Why the previous deploy was broken
1. The GitHub repo was **missing** `dashboard/`, the models, and the dataset — only
   notebooks/src were uploaded, so the app couldn't run.
2. The old root `requirements.txt` pinned ancient, incompatible versions (numpy 1.21.6,
   scikit-learn 1.2.2) and omitted streamlit. It's now pinned to versions that match the
   models (scikit-learn 1.7.1).
3. Data/model paths were resolved against the wrong directory; they're now anchored to
   the script location so they work on Streamlit Cloud.

## Push the correct files
```bash
cd "<this project folder>"
git init
git remote add origin https://github.com/amanparganiha/FDA-10-Multimodal-Credit-Risk-Analysis.git
git add -A           # .gitignore keeps the bloat out
git commit -m "Deploy working dashboard: app, models, dataset, fixed requirements + accurate predictions"
git branch -M main
git push -u origin main --force
```
Streamlit Cloud will redeploy automatically once the push completes.
