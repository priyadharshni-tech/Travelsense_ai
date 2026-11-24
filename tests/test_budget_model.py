import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.budget_model import TransparentBaselineEstimator, TravelBudgetPredictor
from train_models import generate_trip_level_synthetic_data

class TestTravelBudgetPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleaned_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_destinations_cleaned.csv")
        if os.path.exists(cleaned_path):
            cls.df_dest = pd.read_csv(cleaned_path)
        else:
            from src.preprocess import preprocess_travel_data
            pipe_res = preprocess_travel_data()
            cls.df_dest = pipe_res['df_cleaned']

    def test_transparent_baseline_estimator(self):
        """Test rule-based baseline cost calculation and itemized breakdown."""
        sample_row = self.df_dest.iloc[0]
        res = TransparentBaselineEstimator.calculate_baseline_cost(
            sample_row, duration_days=5, group_size=2, accommodation_tier='Standard'
        )

        self.assertIn('baseline_total_inr', res)
        self.assertIn('breakdown', res)
        
        breakdown = res['breakdown']
        self.assertIn('Accommodation', breakdown)
        self.assertIn('Food', breakdown)
        self.assertIn('Local Transport', breakdown)
        self.assertIn('Activities', breakdown)

        sum_breakdown = (
            breakdown['Accommodation'] +
            breakdown['Food'] +
            breakdown['Local Transport'] +
            breakdown['Activities']
        )
        self.assertAlmostEqual(res['baseline_total_inr'], sum_breakdown, places=2)

    def test_random_forest_regressor_train_test_eval(self):
        """Test 80/20 train-test evaluation and metric reporting."""
        train_data = generate_trip_level_synthetic_data(self.df_dest, num_samples=500, random_state=42)
        predictor = TravelBudgetPredictor()
        
        metrics = predictor.fit_and_evaluate(train_data, test_size=0.2, random_state=42)
        
        self.assertEqual(metrics['test_set_size'], 100)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertGreater(metrics['R2'], 0.5)

    def test_prediction_output_structure(self):
        """Test full prediction method output."""
        sample_row = self.df_dest.iloc[0]
        train_data = generate_trip_level_synthetic_data(self.df_dest, num_samples=300, random_state=42)
        
        predictor = TravelBudgetPredictor()
        predictor.fit_and_evaluate(train_data, test_size=0.2, random_state=42)
        
        res = predictor.predict_trip(sample_row, duration_days=4, group_size=2, accommodation_tier='Standard')
        
        self.assertIn('baseline_total_inr', res)
        self.assertIn('ml_predicted_total_inr', res)
        self.assertIn('model_disclaimer', res)
        self.assertIn('breakdown', res)
        self.assertIn('Experimental Demonstration', res['model_disclaimer'])

if __name__ == '__main__':
    unittest.main()
