import os
import sys
import unittest

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recommender import ContentBasedRecommender

class TestContentBasedRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recommender = ContentBasedRecommender()

    def test_recommender_initialization(self):
        """Test dataset loading and feature matrix construction."""
        self.assertIsNotNone(self.recommender.df)
        self.assertGreater(len(self.recommender.df), 0)
        self.assertIsNotNone(self.recommender.feature_matrix)

    def test_hill_station_cold_recommendation(self):
        """Test recommendation for Hill Station + Cold climate + Summer season."""
        recs = self.recommender.recommend(
            total_budget_inr=50000,
            group_size=2,
            duration_days=5,
            destination_type="Hill Station",
            climate="Cold",
            best_season="Summer",
            top_k=5
        )

        self.assertEqual(len(recs), 5)
        # Check required output columns
        required_cols = ['destination', 'state', 'estimated_trip_cost', 'similarity_percent', 'recommendation_reasons']
        for col in required_cols:
            self.assertIn(col, recs.columns)

        # Top recommendation should match Hill Station or Cold climate
        top_row = recs.iloc[0]
        self.assertTrue(
            top_row['destination_type'] == "Hill Station" or top_row['climate'] == "Cold",
            f"Expected Hill Station or Cold climate, got {top_row['destination_type']} / {top_row['climate']}"
        )

    def test_beach_tropical_recommendation(self):
        """Test recommendation for Beach + Tropical climate + Winter season."""
        recs = self.recommender.recommend(
            total_budget_inr=40000,
            group_size=2,
            duration_days=4,
            destination_type="Beach",
            climate="Tropical",
            best_season="Winter",
            top_k=5
        )

        self.assertEqual(len(recs), 5)
        top_row = recs.iloc[0]
        self.assertTrue(
            top_row['destination_type'] == "Beach" or top_row['climate'] == "Tropical",
            f"Expected Beach or Tropical, got {top_row['destination_type']} / {top_row['climate']}"
        )

    def test_different_preferences_yield_different_results(self):
        """Test that different inputs produce different recommendations."""
        recs_hill = self.recommender.recommend(
            total_budget_inr=60000, group_size=2, duration_days=5,
            destination_type="Hill Station", climate="Cold", best_season="Summer"
        )

        recs_heritage = self.recommender.recommend(
            total_budget_inr=60000, group_size=2, duration_days=5,
            destination_type="Heritage/Cultural", climate="Warm", best_season="Winter"
        )

        top_hill_destinations = set(recs_hill['destination'].tolist())
        top_heritage_destinations = set(recs_heritage['destination'].tolist())

        # Ensure top recommendations are distinct
        self.assertNotEqual(top_hill_destinations, top_heritage_destinations)

    def test_trip_cost_calculation(self):
        """Test exact formula for estimated trip cost."""
        sample_row = self.recommender.df.iloc[0]
        cost = self.recommender.calculate_trip_cost(sample_row, group_size=2, duration_days=3)
        
        expected_cost = (
            sample_row['average_hotel_cost'] * 1 * 3 +
            sample_row['average_food_cost'] * 2 * 3 +
            sample_row['average_local_transport_cost'] * 2 * 3 +
            sample_row['average_activity_cost'] * 2
        )
        self.assertAlmostEqual(cost, expected_cost, places=2)

if __name__ == '__main__':
    unittest.main()
