import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "indian_destinations_cleaned.csv")

class ContentBasedRecommender:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.load_data()
        self._build_feature_matrix()

    def load_data(self):
        """
        Loads cleaned Indian destination dataset.
        """
        if not os.path.exists(self.data_path):
            from src.preprocess import preprocess_travel_data
            pipe_res = preprocess_travel_data()
            self.df = pipe_res['df_cleaned']
        else:
            self.df = pd.read_csv(self.data_path)

        # Ensure total daily cost column exists
        if 'total_daily_cost_inr' not in self.df.columns:
            self.df['total_daily_cost_inr'] = (
                self.df['average_hotel_cost'] +
                self.df['average_food_cost'] +
                self.df['average_local_transport_cost'] +
                self.df['average_activity_cost']
            )

        # Fallback image_url if missing
        default_img = "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"
        if 'image_url' not in self.df.columns:
            self.df['image_url'] = default_img
        else:
            self.df['image_url'] = self.df['image_url'].fillna(default_img)

    def _build_feature_matrix(self):
        """
        Builds destination feature matrix using Scikit-Learn One-Hot Encoding and StandardScaler.
        """
        self.cat_cols = ['destination_type', 'climate', 'best_season']
        self.num_cols = [
            'average_hotel_cost', 'average_food_cost',
            'average_local_transport_cost', 'average_activity_cost',
            'popularity_score', 'total_daily_cost_inr'
        ]

        # 1. One-Hot Encoding for categorical features
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = self.ohe.fit_transform(self.df[self.cat_cols])

        # 2. Feature Scaling for numerical features
        self.scaler = StandardScaler()
        num_scaled = self.scaler.fit_transform(self.df[self.num_cols])

        # Combine into complete feature matrix (Weighting categorical features slightly higher)
        self.feature_matrix = np.hstack([cat_encoded * 1.5, num_scaled])

    def calculate_trip_cost(self, row, group_size, duration_days):
        """
        Calculates exact estimated trip cost in INR (₹) for given group size and stay duration.
        """
        rooms_needed = int(np.ceil(group_size / 2.0))
        hotel_total = row['average_hotel_cost'] * rooms_needed * duration_days
        food_total = row['average_food_cost'] * group_size * duration_days
        transport_total = row['average_local_transport_cost'] * group_size * duration_days
        activity_total = row['average_activity_cost'] * group_size

        return round(hotel_total + food_total + transport_total + activity_total, 2)

    def generate_recommendation_reasons(self, row, user_type, user_climate, user_season, trip_cost, budget):
        """
        Generates dynamic explanations for why a destination was recommended.
        """
        reasons = []

        if user_type != "Any" and row['destination_type'].lower() == user_type.lower():
            reasons.append(f"Matches your preferred '{row['destination_type']}' destination type.")
        else:
            reasons.append(f"Popular '{row['destination_type']}' travel spot.")

        if user_climate != "Any" and row['climate'].lower() == user_climate.lower():
            reasons.append(f"Offers your ideal '{row['climate']}' climate.")

        if user_season != "Any" and row['best_season'].lower() == user_season.lower():
            reasons.append(f"Best traveled during '{row['best_season']}'.")

        if trip_cost <= budget:
            savings = round(budget - trip_cost, 2)
            if savings > 0:
                reasons.append(f"Fits within budget (₹{savings:,.0f} remaining).")
            else:
                reasons.append("Exact match for your target budget.")
        else:
            reasons.append("Slightly exceeds strict budget target.")

        if row['popularity_score'] >= 4.5:
            reasons.append(f"Highly rated destination (⭐ {row['popularity_score']}/5.0).")

        return reasons

    def recommend(self, total_budget_inr, group_size, duration_days, destination_type="Any", climate="Any", best_season="Any", top_k=5):
        """
        Recommends top_k destinations matching user preference vector.
        """
        # 1. Create synthetic query dataframe for encoding & scaling
        user_cat_dict = {
            'destination_type': destination_type if destination_type != "Any" else self.df['destination_type'].mode()[0],
            'climate': climate if climate != "Any" else self.df['climate'].mode()[0],
            'best_season': best_season if best_season != "Any" else self.df['best_season'].mode()[0]
        }
        
        user_cat_df = pd.DataFrame([user_cat_dict])
        user_cat_encoded = self.ohe.transform(user_cat_df)

        # Target daily budget per person/room estimate
        target_daily_total = float(total_budget_inr) / max(1, (group_size * duration_days))
        target_hotel = target_daily_total * 0.50
        target_food = target_daily_total * 0.25
        target_transport = target_daily_total * 0.15
        target_activity = target_daily_total * 0.10

        user_num_df = pd.DataFrame([{
            'average_hotel_cost': target_hotel,
            'average_food_cost': target_food,
            'average_local_transport_cost': target_transport,
            'average_activity_cost': target_activity,
            'popularity_score': 4.5,
            'total_daily_cost_inr': target_daily_total
        }])
        
        user_num_scaled = self.scaler.transform(user_num_df)

        # User vector
        user_vector = np.hstack([user_cat_encoded * 1.5, user_num_scaled])

        # 2. Compute Cosine Similarity between User Vector and Destination Matrix
        sim_scores = cosine_similarity(user_vector, self.feature_matrix).flatten()

        results_df = self.df.copy()
        results_df['similarity_score'] = sim_scores

        # 3. Calculate Estimated Trip Cost for all destinations
        trip_costs = []
        for _, row in results_df.iterrows():
            cost = self.calculate_trip_cost(row, group_size, duration_days)
            trip_costs.append(cost)

        results_df['estimated_trip_cost'] = trip_costs

        # 4. Budget Filtering: Filter by total_budget_inr
        within_budget_df = results_df[results_df['estimated_trip_cost'] <= total_budget_inr].copy()

        # If budget is tight and fewer than top_k destinations meet strict limit, include closest options sorted by budget fit
        if len(within_budget_df) < top_k:
            sorted_df = results_df.sort_values(by=['similarity_score', 'estimated_trip_cost'], ascending=[False, True])
        else:
            sorted_df = within_budget_df.sort_values(by='similarity_score', ascending=False)

        # Take Top K
        top_recs = sorted_df.head(top_k).copy()

        # Convert similarity to Percentage (e.g. 70% to 99%)
        min_sim = results_df['similarity_score'].min()
        max_sim = results_df['similarity_score'].max()
        
        if max_sim > min_sim:
            top_recs['similarity_percent'] = (
                70 + 29 * (top_recs['similarity_score'] - min_sim) / (max_sim - min_sim)
            ).round(1)
        else:
            top_recs['similarity_percent'] = 85.0

        # Generate reasons for recommendation
        reasons_list = []
        for _, row in top_recs.iterrows():
            reasons = self.generate_recommendation_reasons(
                row, destination_type, climate, best_season, row['estimated_trip_cost'], total_budget_inr
            )
            reasons_list.append(reasons)

        top_recs['recommendation_reasons'] = reasons_list

        return top_recs
