import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SEED = 42
np.random.seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "indian_destinations_raw.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "indian_destinations_cleaned.csv")

SYNTHETIC_DISCLAIMER = (
    "DISCLAIMER: This dataset contains synthetic prototype data covering Indian travel destinations. "
    "All monetary values (in INR) are generated for machine learning & data science demonstration purposes "
    "and do not represent actual live market rates."
)

def load_raw_dataset(file_path=RAW_DATA_PATH):
    if not os.path.exists(file_path):
        from generate_indian_dataset import generate_indian_dataset
        generate_indian_dataset()
    df = pd.read_csv(file_path)
    return df

def inspect_dataset(df):
    initial_shape = df.shape
    duplicate_count = int(df.duplicated().sum())
    missing_summary = df.isnull().sum().to_dict()
    
    return {
        'total_rows': initial_shape[0],
        'total_columns': initial_shape[1],
        'duplicate_count': duplicate_count,
        'missing_values': missing_summary
    }

def preprocess_travel_data(file_path=RAW_DATA_PATH, save_cleaned=True, seed=SEED):
    np.random.seed(seed)
    df = load_raw_dataset(file_path)
    stats_before = inspect_dataset(df)
    
    # Drop duplicates
    df_cleaned = df.drop_duplicates().copy()
    
    # Handle missing values
    num_cols = [
        'average_hotel_cost', 'average_food_cost',
        'average_local_transport_cost', 'average_activity_cost',
        'recommended_duration', 'popularity_score'
    ]
    
    for col in num_cols:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            group_medians = df_cleaned.groupby('destination_type')[col].transform('median')
            df_cleaned[col] = df_cleaned[col].fillna(group_medians).fillna(df_cleaned[col].median())
            
    # Calculate Total Daily Cost in INR (₹)
    df_cleaned['total_daily_cost_inr'] = (
        df_cleaned['average_hotel_cost'] +
        df_cleaned['average_food_cost'] +
        df_cleaned['average_local_transport_cost'] +
        df_cleaned['average_activity_cost']
    ).round(2)

    # Standardize string formatting
    cat_cols = ['destination', 'state', 'destination_type', 'climate', 'best_season']
    for c in cat_cols:
        if c in df_cleaned.columns:
            df_cleaned[c] = df_cleaned[c].astype(str).str.strip()

    # Fallback image_url if missing
    if 'image_url' not in df_cleaned.columns:
        df_cleaned['image_url'] = "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"
    else:
        df_cleaned['image_url'] = df_cleaned['image_url'].fillna("https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80")

    # Encode & Scale for ML
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_features = ['destination_type', 'climate', 'best_season']
    encoded_cats = ohe.fit_transform(df_cleaned[cat_features])
    encoded_cat_names = ohe.get_feature_names_out(cat_features)
    encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoded_cat_names, index=df_cleaned.index)
    
    scaler = StandardScaler()
    scaled_num_cols = num_cols + ['total_daily_cost_inr']
    scaled_num_data = scaler.fit_transform(df_cleaned[scaled_num_cols])
    scaled_num_names = [f"{col}_scaled" for col in scaled_num_cols]
    scaled_num_df = pd.DataFrame(scaled_num_data, columns=scaled_num_names, index=df_cleaned.index)
    
    df_preprocessed = pd.concat([df_cleaned, scaled_num_df, encoded_cat_df], axis=1)

    stats_after = inspect_dataset(df_cleaned)

    if save_cleaned:
        os.makedirs(DATA_DIR, exist_ok=True)
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
        ml_path = os.path.join(DATA_DIR, "indian_destinations_ml_ready.csv")
        df_preprocessed.to_csv(ml_path, index=False)

    return {
        'df_raw': df,
        'df_cleaned': df_cleaned,
        'df_preprocessed': df_preprocessed,
        'stats_before': stats_before,
        'stats_after': stats_after,
        'disclaimer': SYNTHETIC_DISCLAIMER
    }

if __name__ == "__main__":
    results = preprocess_travel_data()
    print("[SUCCESS] Preprocessing Pipeline Completed Successfully!")
    print(f"Stats Before Cleaning: {results['stats_before']}")
    print(f"Stats After Cleaning:  {results['stats_after']}")
    print(f"\n{results['disclaimer']}")
