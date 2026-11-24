import os
import sys
import unittest
import pandas as pd

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import preprocess_travel_data, load_raw_dataset, inspect_dataset
from src.recommender import ContentBasedRecommender
from src.budget_model import TransparentBaselineEstimator, TravelBudgetPredictor
from src.analytics import (
    get_summary_metrics,
    plot_top_expensive_destinations,
    plot_cost_distribution,
    plot_correlation_heatmap,
    plot_destination_types_pie
)

def run_full_prototype_audit():
    print("==========================================================")
    print("      TRAVELSENSE AI - FULL PROTOTYPE AUDIT REPORT       ")
    print("==========================================================")

    # 1. Dataset & Preprocessing Check
    print("\n[STEP 1] Checking Raw Dataset & Running Preprocessing Pipeline...")
    raw_df = load_raw_dataset()
    print(f"  - Raw dataset loaded successfully: {len(raw_df)} rows")
    
    pipe_res = preprocess_travel_data()
    cleaned_df = pipe_res['df_cleaned']
    print(f"  - Preprocessing completed: {len(cleaned_df)} cleaned unique rows")
    print(f"  - Duplicates before: {pipe_res['stats_before']['duplicate_count']} -> After: {pipe_res['stats_after']['duplicate_count']}")
    print(f"  - Missing values before: {sum(pipe_res['stats_before']['missing_values'].values())} -> After: {sum(pipe_res['stats_after']['missing_values'].values())}")

    # 2. Recommendation Algorithm - 3 User Profiles Test
    print("\n[STEP 2] Testing Content-Based Recommender with 3 User Profiles...")
    recommender = ContentBasedRecommender()

    profiles = [
        {"name": "Profile 1: Alpine Hill Station", "budget": 50000, "group": 2, "days": 5, "type": "Hill Station", "climate": "Cold", "season": "Summer"},
        {"name": "Profile 2: Tropical Beach Retreat", "budget": 40000, "group": 2, "days": 4, "type": "Beach", "climate": "Tropical", "season": "Winter"},
        {"name": "Profile 3: Cultural Heritage Tour", "budget": 60000, "group": 3, "days": 6, "type": "Heritage/Cultural", "climate": "Warm", "season": "Winter"}
    ]

    for p in profiles:
        recs = recommender.recommend(
            total_budget_inr=p["budget"],
            group_size=p["group"],
            duration_days=p["days"],
            destination_type=p["type"],
            climate=p["climate"],
            best_season=p["season"],
            top_k=5
        )
        print(f"\n  --- {p['name']} ---")
        print(f"  Query: Budget=INR {p['budget']:,}, Travelers={p['group']}, Days={p['days']}, Type={p['type']}, Climate={p['climate']}")
        print(f"  Top 3 Recommendations:")
        for rank, (_, row) in enumerate(recs.head(3).iterrows(), 1):
            print(f"    #{rank} {row['destination']} ({row['state']}) | Cost: INR {row['estimated_trip_cost']:,.2f} | Match: {row['similarity_percent']}%")

    # 3. Budget Predictor Engine Test
    print("\n[STEP 3] Testing Budget Prediction Module (Baseline & RandomForest ML)...")
    predictor = TravelBudgetPredictor()
    model_path = os.path.join(os.path.dirname(__file__), "models", "budget_model.joblib")
    if os.path.exists(model_path):
        predictor.load(model_path)
    else:
        from train_models import generate_trip_level_synthetic_data
        dataset_df = generate_trip_level_synthetic_data(cleaned_df, num_samples=1000, random_state=42)
        predictor.fit_and_evaluate(dataset_df, test_size=0.2, random_state=42)

    sample_dest = cleaned_df.iloc[0]
    pred_res = predictor.predict_trip(sample_dest, duration_days=5, group_size=2, accommodation_tier="Standard")
    print(f"  - Target Destination: {sample_dest['destination']} ({sample_dest['state']})")
    print(f"  - Transparent Baseline Total: INR {pred_res['baseline_total_inr']:,.2f}")
    print(f"  - RandomForest ML Prediction: INR {pred_res['ml_predicted_total_inr']:,.2f}")
    print(f"  - Itemized Breakdown: {pred_res['breakdown']}")
    if predictor.metrics:
        print(f"  - Model Metrics on Held-out 80/20 Test Split: MAE=INR {predictor.metrics['MAE']:,.2f}, RMSE=INR {predictor.metrics['RMSE']:,.2f}, R2={predictor.metrics['R2']:.4f}")

    # 4. Dashboard Chart Verification
    print("\n[STEP 4] Verifying Dashboard Chart Generators (Matplotlib / Seaborn)...")
    fig1 = plot_top_expensive_destinations(cleaned_df)
    fig2 = plot_cost_distribution(cleaned_df)
    fig3 = plot_correlation_heatmap(cleaned_df)
    fig4 = plot_destination_types_pie(cleaned_df)
    print("  - All 4 dashboard figures generated successfully without errors.")

    print("\n==========================================================")
    print("      [SUCCESS] PROTOTYPE AUDIT PASSED: ALL MODULES READY!      ")
    print("==========================================================")

if __name__ == "__main__":
    run_full_prototype_audit()
