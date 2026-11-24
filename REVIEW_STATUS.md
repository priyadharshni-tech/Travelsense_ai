# 📜 TravelSense AI – Project Review Status & Prototype Documentation

**Project Name:** TravelSense AI – Intelligent Travel Recommendation & Budget Prediction System  
**Target Domain:** Indian Travel Destinations (200 Destinations across 20+ States)  
**Status:** Working Prototype Verified & Production Ready for College Review  

---

## 1. 🌟 Features Successfully Implemented

- [x] **Data Collection & Preprocessing Pipeline (`src/preprocess.py`)**
  - Deduplication (`205` raw records $\rightarrow$ `200` unique destination records).
  - Group median imputation for missing numerical values (`average_hotel_cost`, `average_food_cost`).
  - Feature engineering for total estimated daily cost in Indian Rupees (`total_daily_cost_inr`).
  - One-Hot encoding of categorical attributes & StandardScaler normalization.
- [x] **Content-Based Recommendation Engine (`src/recommender.py`)**
  - Multi-attribute vectorization using `OneHotEncoder` & `StandardScaler`.
  - Scikit-Learn `cosine_similarity` score calculation.
  - Strict budget eligibility filtering ($\text{Estimated Trip Cost} \le \text{Total Travel Budget}$).
  - Dynamic generation of bullet-point recommendation reasons.
  - Tested against 3 distinct user profiles (Alpine Hill Station, Tropical Beach, Cultural Heritage).
- [x] **Travel Budget Prediction Module (`src/budget_model.py`)**
  - **Transparent Baseline Estimator**: Rule-based cost calculation directly from destination hotel, dining, transit, and activity rate columns.
  - **Random Forest Regressor**: Scikit-Learn `Pipeline` trained on 80/20 train-test split with strict data leakage prevention.
  - Itemized cost breakdown (Accommodation, Food, Local Transport, Activities).
  - Joblib model persistence (`models/budget_model.joblib`).
- [x] **Interactive Analytics Dashboard (`src/analytics.py` & `app.py`)**
  - Summary KPI cards (Total Destinations, Avg Daily Cost, Categories Count).
  - Interactive multiselect / dropdown filters for Destination Type & Climate.
  - **Chart 1**: Top 10 Most Expensive Destinations Bar Chart (with data annotations & factual insights).
  - **Chart 2**: Daily Travel Cost Distribution Histogram (with KDE curve, Mean & Median reference lines).
  - **Chart 3**: Numerical Feature Correlation Matrix Heatmap.
  - **Chart 4**: Destination Categories Proportion Donut / Pie Chart.
- [x] **Streamlit Web Application (`app.py`)**
  - Modern travel-themed UI with custom blue & white styling (`assets/style.css`).
  - 5 interactive tabs covering Home, Analytics, Recommendation Form, Top 5 Recommendations, and Budget Predictor.
- [x] **Automated Unit Test Suite (`tests/`)**
  - `10` passing unit tests covering preprocessing, recommender matching, budget calculations, and dashboard figure generation.

---

## 2. 📊 Dataset Source & Limitations

- **Dataset Path**: [`data/indian_destinations_cleaned.csv`](file:///c:/Users/Priyadharshni%20%20S/OneDrive/Desktop/travelsenseAI/data/indian_destinations_cleaned.csv)
- **Record Count**: `200` unique Indian travel destinations across `20+` Indian states/territories.
- **Monetary Unit**: Indian Rupees (INR / ₹).
- **Dataset Classification**: **Synthetic Prototype Dataset**. All monetary values are synthetically generated for machine learning pipeline demonstration.
- **Limitations**:
  - Does not reflect real-time live market prices or seasonal price surge rates.
  - Flight costs from specific origin cities are estimated as average regional transit baselines.

---

## 3. ⚙️ Algorithms Used

1. **Content-Based Similarity Recommendation**:
   - `OneHotEncoder` for `destination_type`, `climate`, `best_season`.
   - `StandardScaler` for numeric expense components.
   - `cosine_similarity(V_user, X_dest)` for ranking destination compatibility.
2. **Travel Budget Prediction**:
   - **Baseline Estimator**: Transparent rate-based multiplication formula.
   - **Machine Learning**: `RandomForestRegressor` with 100 estimators, max depth 12.

---

## 4. 📈 Empirical Model Evaluation Results

Evaluated on a held-out **80/20 Train-Test Split** (`400` test samples):

| Metric | Metric Name | Value (INR ₹) | Status |
| :--- | :--- | :--- | :--- |
| **MAE** | Mean Absolute Error | **₹15,673.93** | Passed |
| **RMSE** | Root Mean Squared Error | **₹27,048.06** | Passed |
| **$R^2$** | R-Squared | **0.8347** | Passed |

---

## 5. 🔮 Features Under Development / Future Scope

- [ ] **Live Flight API Integration**: Fetch real-time flight quotes from Amadeus / Skyscanner APIs.
- [ ] **User Auth & Itinerary Saving**: User login with SQLite / PostgreSQL backend to bookmark trip plans.
- [ ] **Multi-Currency Support**: Dynamic currency converter (INR $\leftrightarrow$ USD, EUR, GBP).
- [ ] **Collaborative Filtering**: Hybrid recommender incorporating user rating interactions.
