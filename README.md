# Interactive ML Evaluation Dashboard

A standalone Streamlit app that trains a scikit-learn Logistic Regression baseline on the **public digits dataset** and displays interactive evaluation visualizations.

## Features
- Stratified train/test split and training-only feature scaling.
- Held-out accuracy and macro-F1.
- Confusion matrix, class-wise metrics, digit image explorer, prediction probabilities.
- Change the random seed to explore sensitivity to the split.

## Run
```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```
Streamlit opens a local browser window, often at `http://localhost:8501`.

## Reproducibility and limitations
The classifier is trained locally using scikit-learn's bundled digits dataset. The app does not store user data or call external APIs. The dashboard is for learning and demonstration; it is not a deployed medical or production system. Results depend on the selected split and do not establish robustness under domain shift.

## License
MIT.
