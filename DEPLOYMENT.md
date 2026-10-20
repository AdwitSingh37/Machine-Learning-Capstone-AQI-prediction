# Run and publish AirLens

## Live deployment

Public URL: **https://airlens-aqi-capstone.streamlit.app/**

Hosted on Streamlit Community Cloud from GitHub branch `main`,
entrypoint `app.py`, Python 3.12. Public sharing is enabled. The hosted app was
verified with the bundled Amaravati sample, which returns AQI **187.5** and
category **Moderate**. Use the public URL in the review submission.

## Run locally

Use Python 3.12. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. The trained artifact is included in
`models/aqi_pipeline.joblib`, so the server does not train at startup.
Only load the model artifact produced by this repository's training script.

## Public deployment: Streamlit Community Cloud

1. Commit and push `app.py`, `aqi_model.py`, `requirements.txt`, `.streamlit/config.toml`, and **the `models/` directory** to the GitHub repository. Include `train_model.py` and the documentation for reproducibility.
2. Sign in at https://share.streamlit.io/ using your GitHub account.
3. Choose **Create app** and deploy from an existing repository.
4. Set repository to `AdwitSingh37/Machine-Learning-Capstone-AQI-prediction`, select the branch containing these files, and set the entrypoint to **app.py**.
5. Under advanced settings choose **Python 3.12**, then deploy.
6. Open the generated `https://…streamlit.app` address and submit a sample prediction. Verify the app's sharing setting allows public access, then open the URL in a signed-out browser.
7. Add the actual working URL and a screenshot of a prediction to your review submission. A local URL or deployment configuration alone does not satisfy the public deployment requirement.

No API keys or external data services are needed. The app loads the bundled trained model and uses user-supplied readings. Do not commit `.streamlit/secrets.toml`.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Alternative: Render

The included `render.yaml` defines a Python web service. Connect the GitHub repository to Render and create a Blueprint from it. The service installs `requirements.txt` and starts Streamlit on Render's assigned port. Choose the free plan when available. Free services may sleep when idle. Deployment still requires a signed-in hosting account and the files pushed to GitHub.

## Retrain and verify

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Retraining replaces the model and `models/metrics.json`. Commit both together. The scikit-learn version is pinned to the version used to serialize the model; retrain if changing it.

The model follows the regression notebook's preprocessing and Random Forest approach, with maximum depth 20 and minimum leaf size 2 for a smaller deployment artifact. All preprocessing is fitted inside the pipeline on the training split. Metrics use a random 80:20 held-out split of labeled historical observations, not an evaluation of future forecasting or unseen cities.
