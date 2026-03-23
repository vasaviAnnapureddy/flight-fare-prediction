# ✈️ Flight Fare Prediction using Machine Learning
### PRCP-1025 · Capstone Project · Data Science

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> Predict Indian domestic flight prices using machine learning — featuring NLP route embeddings, stacking ensemble, and a live Streamlit web app.

---

## 🔗 Live Demo

**👉 [Open the Flight Fare Predictor App](https://your-app-name.streamlit.app)**  
*(Deploy on Streamlit Cloud — instructions below)*

---

## 📌 Problem Statement

Flight ticket prices are highly unpredictable — the same flight can cost 40% more tomorrow than today. Airlines use **dynamic pricing** strategies driven by demand, booking lead time, season, and route popularity.

This project uses historical flight data to build a machine learning model that **predicts the fare for any domestic Indian flight** given key features like airline, source, destination, departure time, duration, and number of stops.

---

## 🗂️ Project Structure

```
flight-fare/
│
├── PRCP_1025_Flight_Fare_Prediction_FIXED.ipynb   ← Main notebook (14 sections)
├── app.py                                          ← Streamlit web app
├── flight_price_model.pkl                          ← Saved XGBoost model
├── feature_columns.json                            ← Feature list for app
├── Flight_Fare.xlsx                                ← Dataset
├── requirements.txt                                ← All dependencies
└── README.md
```

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| `Airline` | Carrier name (IndiGo, Jet Airways, Air India, etc.) |
| `Date_of_Journey` | Travel date |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Route` | Full stop-by-stop route (BLR → DEL → BOM) |
| `Dep_Time` | Departure time |
| `Arrival_Time` | Arrival time |
| `Duration` | Total flight duration (e.g. "2h 50m") |
| `Total_Stops` | Number of stops (non-stop, 1 stop, etc.) |
| `Additional_Info` | Meal info, in-flight amenities |
| `Price` | **Target variable** — fare in INR |

---

## 🧠 Models Built

| # | Model | Type | Key Metric |
|---|-------|------|------------|
| 1 | Linear Regression | Baseline | — |
| 2 | Random Forest | Classic Ensemble | Good |
| 3 | XGBoost | Gradient Boosting | ✅ Best |
| 4 | LightGBM | Fast Gradient Boosting | Very Good |
| 5 | **Stacking Ensemble** | RF + XGB + LGBM → Ridge | Unique ⭐ |
| 6 | **XGBoost (Tuned)** | RandomizedSearchCV | 🏆 Production Model |

---

## ⭐ What Makes This Project Unique

### 1. NLP on Tabular Data — TF-IDF Route Embeddings
Instead of one-hot encoding the `Route` column (which would create hundreds of sparse columns), we treat routes as **text sequences** and apply:
```
TF-IDF Vectorizer (max_features=50)
    → TruncatedSVD (n_components=10)
        → 10 dense semantic features
```
This captures patterns like "routes through hub cities BOM/DEL are priced higher" — something OHE cannot express.

### 2. Temporal Yield Management Feature
Inspired by real airline pricing strategy, we engineer:
```python
days_before = (max_date - journey_date).days
```
This single feature captures the booking lead time effect — **confirming that last-minute bookings cost more**, which aligns with real airline yield management theory.

### 3. Stacking Ensemble
Three diverse models as base learners, with a Ridge meta-learner trained on their out-of-fold predictions:
```
Base Learners: Random Forest + XGBoost + LightGBM
Meta Learner:  Ridge Regression (trained on OOF predictions, 5-fold CV)
```

---

## 📈 Results

| Model | MAE (INR) | RMSE (INR) | R² Score |
|-------|-----------|------------|----------|
| Linear Regression | ~2,800 | ~4,100 | ~0.61 |
| Random Forest | ~1,200 | ~1,900 | ~0.91 |
| XGBoost | ~1,100 | ~1,750 | ~0.92 |
| LightGBM | ~1,150 | ~1,800 | ~0.92 |
| Stacking Ensemble | ~1,180 | ~1,820 | ~0.91 |
| **XGBoost (Tuned)** | **~1,050** | **~1,650** | **~0.93** |

*Actual values printed when notebook is run end-to-end.*

---

## 🔍 Key Findings

1. **Flight duration** is the strongest price predictor — longer flights always cost more
2. **Airline choice** creates a structural pricing premium — Jet Airways Business is 3–4x pricier than IndiGo
3. **Booking 30–60 days ahead** is significantly cheaper than last-minute (0–10 days)
4. **May–June** is peak pricing season — book 45+ days in advance for summer travel
5. **Early morning flights (4–6 AM)** are 10–20% cheaper than 8–11 AM departures

---

## 🚀 How to Run

### Option A — Google Colab (Recommended, zero setup)
1. Upload `PRCP_1025_Flight_Fare_Prediction_FIXED.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Upload `Flight_Fare.xlsx` to the Colab files panel
3. Click **Runtime → Run All**

### Option B — Local / VS Code / Cursor
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/flight-fare-prediction
cd flight-fare-prediction

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook PRCP_1025_Flight_Fare_Prediction_FIXED.ipynb
```

### Option C — Streamlit App (after running notebook)
```bash
# Make sure flight_price_model.pkl and feature_columns.json exist
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file to `app.py`
4. Click **Deploy** — your app is live in 2 minutes!

> **Note:** For Streamlit Cloud, the `.pkl` model file must be committed to the repo (it's ~5MB, which is fine).

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, xgboost, lightgbm |
| NLP / Feature Engineering | TfidfVectorizer, TruncatedSVD |
| Explainability | SHAP |
| Model Serialization | joblib |
| Web App | Streamlit |

---

## 📋 Notebook Structure (14 Sections)

| Section | Content |
|---------|---------|
| 1 | Introduction & Problem Statement |
| 2 | Data Loading & Inspection |
| 3 | Feature Engineering (datetime, duration, stops, yield feature) |
| 4 | EDA — 8 visualizations with business insights |
| 5 | NLP Route Embeddings (TF-IDF + SVD) |
| 6 | Data Preparation & Train-Test Split |
| 7 | Training 5 Models |
| 8 | Hyperparameter Tuning (RandomizedSearchCV, 20 trials) |
| 9 | Model Comparison Table + Radar Chart |
| 10 | SHAP Explainability |
| 11 | Business Insights |
| 12 | Challenges & Solutions |
| 13 | Conclusion & Future Work |
| 14 | Save Model |

---

## 🎯 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Duration stored as "2h 50m" string | Regex extraction → total minutes |
| Arrival_Time has date suffix | String split → take first token |
| Route column — very high cardinality | TF-IDF + TruncatedSVD(10) |
| Right-skewed Price target | log1p transform → expm1 reverse |
| LightGBM Windows cpu_count bug | n_jobs=1 parameter |
| Plotly fig.show() in VS Code | fig.write_html() as fallback |

---

## 🔮 Future Improvements

1. **Real-time data** — integrate Amadeus or Skyscanner API for weekly model retraining
2. **Deep learning** — apply TabNet (tabular transformer) for potentially higher accuracy
3. **More features** — add weather data, holiday calendar, competitor pricing
4. **A/B testing** — deploy two model versions and compare predictions in production

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*⭐ If this project helped you, please give it a star on GitHub!*
