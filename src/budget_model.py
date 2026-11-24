import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ACCOMMODATION_MULTIPLIERS = {
    'Budget': 0.70,
    'Standard': 1.00,
    'Luxury': 1.80,
    'Ultra Luxury': 2.60
}

class TransparentBaselineEstimator:
    """
    Rule-based transparent baseline cost estimator using direct destination daily cost rates.
    """
    @staticmethod
    def calculate_baseline_cost(dest_row, duration_days, group_size, accommodation_tier='Standard'):
        """
        Calculates transparent baseline costs from primary rate columns:
        - average_hotel_cost
        - average_food_cost
        - average_local_transport_cost
        - average_activity_cost
        """
        acc_mult = ACCOMMODATION_MULTIPLIERS.get(accommodation_tier, 1.0)
        rooms_needed = int(np.ceil(group_size / 2.0))

        hotel_total = round(dest_row['average_hotel_cost'] * rooms_needed * duration_days * acc_mult, 2)
        food_total = round(dest_row['average_food_cost'] * group_size * duration_days, 2)
        transport_total = round(dest_row['average_local_transport_cost'] * group_size * duration_days, 2)
        activity_total = round(dest_row['average_activity_cost'] * group_size, 2)

        baseline_total = round(hotel_total + food_total + transport_total + activity_total, 2)

        return {
            'baseline_total_inr': baseline_total,
            'breakdown': {
                'Accommodation': hotel_total,
                'Food': food_total,
                'Local Transport': transport_total,
                'Activities': activity_total
            }
        }


class TravelBudgetPredictor:
    """
    Machine Learning Budget Predictor using RandomForestRegressor.
    Data Leakage Prevention: Does NOT take total cost or direct sum components as input features.
    Input Features available at prediction time:
    - state, destination_type, climate, best_season, popularity_score
    - duration_days, group_size, accommodation_tier
    """
    def __init__(self):
        self.model_status = "Experimental Demonstration (Trained on synthetic prototype labels)"
        self.feature_cols = [
            'state', 'destination_type', 'climate', 'best_season',
            'popularity_score', 'duration_days', 'group_size', 'accommodation_tier'
        ]
        self.metrics = {}
        self._build_pipeline()

    def _build_pipeline(self):
        numeric_features = ['popularity_score', 'duration_days', 'group_size']
        categorical_features = ['state', 'destination_type', 'climate', 'best_season', 'accommodation_tier']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))
        ])

    def fit_and_evaluate(self, dataset_df, test_size=0.2, random_state=42):
        """
        Trains RandomForestRegressor using an 80/20 train-test split.
        Returns MAE, RMSE, and R2 on held-out test data.
        """
        X = dataset_df[self.feature_cols]
        y = dataset_df['total_budget_inr']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.pipeline.fit(X_train, y_train)

        preds = self.pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(root_mean_squared_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        self.metrics = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'test_set_size': len(X_test)
        }
        return self.metrics

    def predict_trip(self, dest_row, duration_days, group_size, accommodation_tier='Standard'):
        """
        Calculates both Transparent Baseline Cost and ML Model Prediction.
        """
        # 1. Transparent Baseline Estimation
        baseline_res = TransparentBaselineEstimator.calculate_baseline_cost(
            dest_row, duration_days, group_size, accommodation_tier
        )

        # 2. Machine Learning Model Prediction
        input_df = pd.DataFrame([{
            'state': dest_row['state'],
            'destination_type': dest_row['destination_type'],
            'climate': dest_row['climate'],
            'best_season': dest_row['best_season'],
            'popularity_score': dest_row['popularity_score'],
            'duration_days': duration_days,
            'group_size': group_size,
            'accommodation_tier': accommodation_tier
        }])

        ml_predicted_inr = float(self.pipeline.predict(input_df)[0])

        return {
            'baseline_total_inr': baseline_res['baseline_total_inr'],
            'ml_predicted_total_inr': round(ml_predicted_inr, 2),
            'model_disclaimer': self.model_status,
            'metrics': self.metrics,
            'breakdown': baseline_res['breakdown']
        }

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump({'pipeline': self.pipeline, 'metrics': self.metrics}, file_path)

    def load(self, file_path):
        data = joblib.load(file_path)
        if isinstance(data, dict) and 'pipeline' in data:
            self.pipeline = data['pipeline']
            self.metrics = data.get('metrics', {})
        else:
            self.pipeline = data
        return self
