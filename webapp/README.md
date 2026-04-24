# 36-Month Web Calculator

This folder contains the minimal Streamlit front end for the 36-month MCI-to-AD calculator.

## Files

- `model_runtime.py`
  - Loads the exported model artifact and runs scoring.
- `minimal_app.py`
  - The only Streamlit page entry point.
- `requirements.txt`
  - Minimal dependency list for the web UI.

## Artifact Export

Export the deployable model artifact with:

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' 网页计算器\code\export_final_model_36m_web_calculator.py
```

This writes the serialized artifact to:

- `网页计算器/artifacts/final_model_36m_model2_lr.joblib`
- `网页计算器/artifacts/final_model_36m_model2_lr_metadata.json`

## Streamlit Launch

If `streamlit` is not yet installed in the `dlnlp` environment, install it first:

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' -m pip install streamlit
```

Then launch the page:

```powershell
& 'C:\Users\HuifangLiu\miniconda3\envs\dlnlp\python.exe' -m streamlit run 网页计算器\webapp\minimal_app.py
```

This version loads the exported artifact from the local `artifacts/` folder and does not depend on the original training tables at runtime.

## Docker Deployment

Build the container image from the repository root:

```powershell
docker build -f 网页计算器\webapp\Dockerfile -t adni-risk-calculator .
```

Run the container:

```powershell
docker run --rm -p 8501:8501 adni-risk-calculator
```

The app will be available at `http://localhost:8501`.

## Scope Note

This prototype follows the ADNI internal final-model definition. It is useful for manuscript-aligned demonstration and method presentation. The published NACC external analysis used a different common-clinical model because `ADAS-Cog13` was unavailable in the current NACC extract.

## Batch Scoring

The minimal page includes a batch CSV expander. Missing columns fall back to the training defaults, and scored results can be downloaded as CSV.
