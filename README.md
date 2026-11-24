# ✈️ TravelSense AI – Intelligent Travel Recommendation & Budget Prediction System

**TravelSense AI** is a data science and machine learning web application built with Python and Streamlit. It delivers personalized travel destination recommendations using content-based similarity filtering and predicts detailed travel budgets with a trained Scikit-Learn Machine Learning model.

---

## 🌟 Key Features

1. **🏠 Home Dashboard**
   - Sleek travel-themed UI with blue & white color palette.
   - Live system metrics (Total destinations tracked, continental coverage, model R² score, average safety rating).
   - Core system capabilities breakdown and featured destination showcase.

2. **🧳 Travel Preference Input Form**
   - Customizable preference filters: Primary Travel Style, Max Daily Budget ($), Preferred Climate, Best Season, Continent, Minimum Safety Index, and activity tags.

3. **🌟 Destination Recommendations**
   - Content-Based Recommendation algorithm (TF-IDF vectorizer + Cosine Similarity + Budget-fit scoring).
   - Ranked destination cards featuring match percentage, daily cost, flight estimates, safety indices, climate details, and one-click budget calculation transfer.

4. **💰 Travel Budget Prediction**
   - Machine Learning regression model (`RandomForestRegressor`) estimating total trip expenses.
   - Factors in destination cost bases, trip duration, group sizes, accommodation style, activity levels, and seasonal multipliers.
   - Itemized cost breakdown pie charts (Flight/Transport, Lodging, Food, Activities, Contingency) with 95% confidence intervals and actionable savings recommendations.

5. **📊 Travel Analytics Dashboard**
   - Interactive exploratory data visualizer powered by Plotly & Seaborn.
   - Cost distributions across travel styles & continents, Safety vs. Popularity bubble matrix, and searchable dataset table with CSV export.

---

## 📁 Project Folder Structure

```text
travelsenseAI/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # System documentation
├── train_models.py             # Script to generate synthetic data & fit ML model
├── data/
│   └── destinations.csv        # Dataset of 50 worldwide destinations
├── notebooks/
│   └── eda.ipynb               # Jupyter Notebook for EDA & model validation
├── src/
│   ├── preprocess.py           # Data loading & synthetic dataset generator
│   ├── recommender.py          # TF-IDF + Cosine Similarity Recommender Engine
│   └── budget_model.py         # RandomForest Regression Budget Predictor Pipeline
├── models/
│   └── budget_model.joblib     # Pre-trained ML model pipeline artifact
└── assets/
    └── style.css               # Modern blue & white UI stylesheet
```

---

## 🚀 Quick Start & Installation

### 1. Clone or Open Project Workspace
Open your terminal in the project directory:
```bash
cd travelsenseAI
```

### 2. Install Required Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train Machine Learning Models
To generate fresh synthetic training samples and re-fit the ML pipeline:
```bash
python train_models.py
```

### 4. Launch the Web Application
Run the Streamlit app:
```bash
streamlit run app.py
```

After running the command, your default browser will open automatically at `http://localhost:8501`.

---

## 🔬 Machine Learning Architecture

### Recommendation System (`src/recommender.py`)
- **Text Feature Vectorization:** Combines travel styles, secondary styles, climate, season, activities, continent, and descriptive text into a TF-IDF feature matrix.
- **Cosine Similarity:** Computes text relevance vectors comparing user query preferences against all destinations.
- **Composite Scoring:** Combines similarity scores ($55\%$), budget fit penalty/reward ($25\%$), and normalized popularity ($20\%$) to produce a normalized 0-100% Match Percentage.

### Budget Prediction Model (`src/budget_model.py`)
- **Pipeline:** `ColumnTransformer` with `StandardScaler` for numeric values (`avg_daily_cost_usd`, `flight_cost_est_usd`, `duration_days`, `group_size`) and `OneHotEncoder` for categorical attributes (`continent`, `accommodation_style`, `activity_level`, `season`).
- **Regressor:** `RandomForestRegressor` with 100 decision trees.
- **Metrics Achieved:**
  - **R² Score:** `~0.94 - 0.96`
  - **MAE:** `$720` (on multi-week group travel predictions)

---

## 🛠️ Technology Stack
- **Language:** Python 3.9+
- **Frontend / Dashboard UI:** Streamlit, Custom CSS (Vanilla Glassmorphic design)
- **Data Wrangling:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (`TfidfVectorizer`, `RandomForestRegressor`, `ColumnTransformer`, `Pipeline`)
- **Visualization:** Plotly Express, Seaborn, Matplotlib
- **Model Storage:** Joblib
