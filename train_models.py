import os
import numpy as np
import pandas as pd
from src.preprocess import preprocess_travel_data
from src.budget_model import TravelBudgetPredictor, ACCOMMODATION_MULTIPLIERS

def generate_trip_level_synthetic_data(df_cleaned, num_samples=2000, random_state=42):
    """
    Generates synthetic trip-level records based on Indian destination cost statistics.
    Ensures features used at prediction time do not cause data leakage.
    """
    np.random.seed(random_state)
    records = []
    
    tiers = ['Budget', 'Standard', 'Luxury', 'Ultra Luxury']

    for _ in range(num_samples):
        dest = df_cleaned.sample(1).iloc[0]
        duration = int(np.random.randint(2, 16))
        group_size = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.25, 0.45, 0.10, 0.12, 0.05, 0.03]))
        acc_tier = np.random.choice(tiers, p=[0.35, 0.45, 0.15, 0.05])

        acc_mult = ACCOMMODATION_MULTIPLIERS[acc_tier]
        rooms_needed = int(np.ceil(group_size / 2.0))

        hotel_total = dest['average_hotel_cost'] * rooms_needed * duration * acc_mult * np.random.uniform(0.93, 1.07)
        food_total = dest['average_food_cost'] * group_size * duration * np.random.uniform(0.92, 1.08)
        transport_total = dest['average_local_transport_cost'] * group_size * duration * np.random.uniform(0.90, 1.10)
        activity_total = dest['average_activity_cost'] * group_size * np.random.uniform(0.88, 1.12)

        total_budget_inr = round(hotel_total + food_total + transport_total + activity_total, 2)

        records.append({
            'state': dest['state'],
            'destination_type': dest['destination_type'],
            'climate': dest['climate'],
            'best_season': dest['best_season'],
            'popularity_score': dest['popularity_score'],
            'duration_days': duration,
            'group_size': group_size,
            'accommodation_tier': acc_tier,
            'total_budget_inr': total_budget_inr
        })

    return pd.DataFrame(records)

def train_and_save():
    print("[1/3] Loading cleaned dataset...")
    pipe_res = preprocess_travel_data()
    df_cleaned = pipe_res['df_cleaned']

    print("[2/3] Generating trip-level prototype records for ML training (80/20 train-test split)...")
    dataset_df = generate_trip_level_synthetic_data(df_cleaned, num_samples=2000, random_state=42)

    predictor = TravelBudgetPredictor()
    metrics = predictor.fit_and_evaluate(dataset_df, test_size=0.2, random_state=42)

    print("\n--- RANDOM FOREST REGRESSOR EVALUATION (HELD-OUT TEST SET) ---")
    print(f"  Test Set Size : {metrics['test_set_size']} samples")
    print(f"  MAE (Mean Absolute Error) : INR {metrics['MAE']:,.2f}")
    print(f"  RMSE (Root Mean Sq Error) : INR {metrics['RMSE']:,.2f}")
    print(f"  R-Squared (R2)            : {metrics['R2']:.4f}")
    print("-----------------------------------------------------------------")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "budget_model.joblib")

    predictor.save(model_path)
    print(f"[3/3] Model saved with Joblib to {model_path}\n")

if __name__ == "__main__":
    train_and_save()
