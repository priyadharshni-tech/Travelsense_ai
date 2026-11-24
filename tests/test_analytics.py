import os
import sys
import unittest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analytics import (
    get_summary_metrics,
    plot_top_expensive_destinations,
    plot_cost_distribution,
    plot_correlation_heatmap,
    plot_destination_types_pie
)

class TestAnalyticsModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleaned_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_destinations_cleaned.csv")
        if os.path.exists(cleaned_path):
            cls.df = pd.read_csv(cleaned_path)
        else:
            from src.preprocess import preprocess_travel_data
            pipe_res = preprocess_travel_data()
            cls.df = pipe_res['df_cleaned']

    def test_summary_metrics(self):
        metrics = get_summary_metrics(self.df)
        self.assertIn('total_destinations', metrics)
        self.assertIn('avg_daily_cost_inr', metrics)
        self.assertIn('num_categories', metrics)
        self.assertGreater(metrics['total_destinations'], 0)
        self.assertGreater(metrics['avg_daily_cost_inr'], 0)

    def test_plot_generation(self):
        fig1 = plot_top_expensive_destinations(self.df)
        self.assertIsNotNone(fig1)

        fig2 = plot_cost_distribution(self.df)
        self.assertIsNotNone(fig2)

        fig3 = plot_correlation_heatmap(self.df)
        self.assertIsNotNone(fig3)

        fig4 = plot_destination_types_pie(self.df)
        self.assertIsNotNone(fig4)

if __name__ == '__main__':
    unittest.main()
