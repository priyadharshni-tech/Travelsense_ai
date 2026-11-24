import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Dictionary of high-resolution Unsplash images mapped to Indian states/destinations
IMAGE_MAP = {
    "Rajasthan": "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80",
    "Udaipur": "https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=800&q=80",
    "Kerala": "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?auto=format&fit=crop&w=800&q=80",
    "Munnar": "https://images.unsplash.com/photo-1593693397690-362cb9666fc2?auto=format&fit=crop&w=800&q=80",
    "Himachal Pradesh": "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?auto=format&fit=crop&w=800&q=80",
    "Shimla": "https://images.unsplash.com/photo-1562979314-bee7453e911c?auto=format&fit=crop&w=800&q=80",
    "Uttarakhand": "https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=800&q=80",
    "Goa": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=800&q=80",
    "Jammu & Kashmir": "https://images.unsplash.com/photo-1566838217578-19035ee8b2cf?auto=format&fit=crop&w=800&q=80",
    "Ladakh": "https://images.unsplash.com/photo-1581793745862-99fde7fa73d2?auto=format&fit=crop&w=800&q=80",
    "Sikkim": "https://images.unsplash.com/photo-1544735716-392fe2489ffa?auto=format&fit=crop&w=800&q=80",
    "Meghalaya": "https://images.unsplash.com/photo-1541414779316-956a5f3d6b16?auto=format&fit=crop&w=800&q=80",
    "Tamil Nadu": "https://images.unsplash.com/photo-1582510003544-4d00b7f74220?auto=format&fit=crop&w=800&q=80",
    "Karnataka": "https://images.unsplash.com/photo-1600100397608-f010e423b971?auto=format&fit=crop&w=800&q=80",
    "Maharashtra": "https://images.unsplash.com/photo-1567157577867-05ccb1388e66?auto=format&fit=crop&w=800&q=80",
    "West Bengal": "https://images.unsplash.com/photo-1558431382-27e303142255?auto=format&fit=crop&w=800&q=80",
    "Andaman & Nicobar Islands": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80",
    "Gujarat": "https://images.unsplash.com/photo-1609949279531-cf48d64bed89?auto=format&fit=crop&w=800&q=80",
    "Odisha": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=800&q=80",
    "Madhya Pradesh": "https://images.unsplash.com/photo-1600100397608-f010e423b971?auto=format&fit=crop&w=800&q=80",
    "Assam": "https://images.unsplash.com/photo-1516426122078-c23e76319801?auto=format&fit=crop&w=800&q=80",
    "Punjab": "https://images.unsplash.com/photo-1584551246679-0daf3d275d0f?auto=format&fit=crop&w=800&q=80",
    "Uttar Pradesh": "https://images.unsplash.com/photo-1564507592333-c60657eea523?auto=format&fit=crop&w=800&q=80",
    "Varanasi": "https://images.unsplash.com/photo-1561361513-2d000a50f0dc?auto=format&fit=crop&w=800&q=80"
}

states_and_cities = [
    ("Jaipur", "Rajasthan", "Heritage/Cultural", "Warm", "Winter"),
    ("Udaipur", "Rajasthan", "Heritage/Cultural", "Warm", "Winter"),
    ("Jaisalmer", "Rajasthan", "Adventure", "Arid", "Winter"),
    ("Jodhpur", "Rajasthan", "Heritage/Cultural", "Warm", "Winter"),
    ("Pushkar", "Rajasthan", "Religious/Spiritual", "Warm", "Winter"),
    ("Munnar", "Kerala", "Hill Station", "Mild", "Winter"),
    ("Alleppey", "Kerala", "Nature", "Tropical", "Winter"),
    ("Varkala", "Kerala", "Beach", "Tropical", "Winter"),
    ("Kochi", "Kerala", "Heritage/Cultural", "Tropical", "Winter"),
    ("Wayanad", "Kerala", "Nature", "Mild", "Autumn"),
    ("Manali", "Himachal Pradesh", "Hill Station", "Cold", "Summer"),
    ("Shimla", "Himachal Pradesh", "Hill Station", "Cold", "Summer"),
    ("Dharamshala", "Himachal Pradesh", "Religious/Spiritual", "Mild", "Spring"),
    ("Spiti Valley", "Himachal Pradesh", "Adventure", "Cold", "Summer"),
    ("Kasol", "Himachal Pradesh", "Adventure", "Mild", "Summer"),
    ("Rishikesh", "Uttarakhand", "Adventure", "Mild", "Spring"),
    ("Nainital", "Uttarakhand", "Hill Station", "Cold", "Summer"),
    ("Mussoorie", "Uttarakhand", "Hill Station", "Cold", "Summer"),
    ("Kedarnath", "Uttarakhand", "Religious/Spiritual", "Cold", "Summer"),
    ("Auli", "Uttarakhand", "Adventure", "Cold", "Winter"),
    ("North Goa", "Goa", "Beach", "Tropical", "Winter"),
    ("South Goa", "Goa", "Beach", "Tropical", "Winter"),
    ("Panaji", "Goa", "Heritage/Cultural", "Tropical", "Winter"),
    ("Srinagar", "Jammu & Kashmir", "Hill Station", "Cold", "Spring"),
    ("Gulmarg", "Jammu & Kashmir", "Adventure", "Cold", "Winter"),
    ("Pahalgam", "Jammu & Kashmir", "Nature", "Cold", "Summer"),
    ("Leh", "Ladakh", "Adventure", "Cold", "Summer"),
    ("Nubra Valley", "Ladakh", "Adventure", "Cold", "Summer"),
    ("Pangong Tso", "Ladakh", "Nature", "Cold", "Summer"),
    ("Gangtok", "Sikkim", "Hill Station", "Cold", "Spring"),
    ("Pelling", "Sikkim", "Nature", "Cold", "Spring"),
    ("Shillong", "Meghalaya", "Nature", "Mild", "Autumn"),
    ("Cherrapunji", "Meghalaya", "Nature", "Mild", "Monsoon"),
    ("Ooty", "Tamil Nadu", "Hill Station", "Mild", "Summer"),
    ("Kodaikanal", "Tamil Nadu", "Hill Station", "Mild", "Summer"),
    ("Rameswaram", "Tamil Nadu", "Religious/Spiritual", "Warm", "Winter"),
    ("Mahabalipuram", "Tamil Nadu", "Heritage/Cultural", "Tropical", "Winter"),
    ("Coorg", "Karnataka", "Nature", "Mild", "Winter"),
    ("Hampi", "Karnataka", "Heritage/Cultural", "Warm", "Winter"),
    ("Gokarna", "Karnataka", "Beach", "Tropical", "Winter"),
    ("Mysore", "Karnataka", "Heritage/Cultural", "Mild", "Winter"),
    ("Lonavala", "Maharashtra", "Hill Station", "Mild", "Monsoon"),
    ("Mahabaleshwar", "Maharashtra", "Hill Station", "Mild", "Winter"),
    ("Mumbai", "Maharashtra", "Heritage/Cultural", "Tropical", "Winter"),
    ("Darjeeling", "West Bengal", "Hill Station", "Cold", "Spring"),
    ("Kolkata", "West Bengal", "Heritage/Cultural", "Warm", "Winter"),
    ("Havelock Island", "Andaman & Nicobar Islands", "Beach", "Tropical", "Winter"),
    ("Neil Island", "Andaman & Nicobar Islands", "Beach", "Tropical", "Winter"),
    ("Gir National Park", "Gujarat", "Wildlife/Nature", "Warm", "Winter"),
    ("Rann of Kutch", "Gujarat", "Heritage/Cultural", "Warm", "Winter"),
    ("Puri", "Odisha", "Religious/Spiritual", "Tropical", "Winter"),
    ("Khajuraho", "Madhya Pradesh", "Heritage/Cultural", "Warm", "Winter"),
    ("Kaziranga", "Assam", "Wildlife/Nature", "Mild", "Winter"),
    ("Amritsar", "Punjab", "Religious/Spiritual", "Mild", "Winter"),
    ("Varanasi", "Uttar Pradesh", "Religious/Spiritual", "Warm", "Winter"),
    ("Agra", "Uttar Pradesh", "Heritage/Cultural", "Warm", "Winter")
]

records = []
for i in range(200):
    base_dest, base_state, base_type, base_climate, base_season = states_and_cities[i % len(states_and_cities)]
    
    if i >= len(states_and_cities):
        dest_name = f"{base_dest} Zone {i // len(states_and_cities) + 1}"
    else:
        dest_name = base_dest
        
    hotel_cost = round(float(np.random.normal(3500, 1200)), -2)
    hotel_cost = max(800.0, min(18000.0, hotel_cost))

    food_cost = round(float(np.random.normal(900, 300)), -1)
    food_cost = max(250.0, min(4000.0, food_cost))

    transport_cost = round(float(np.random.normal(600, 200)), -1)
    transport_cost = max(150.0, min(2500.0, transport_cost))

    activity_cost = round(float(np.random.normal(800, 350)), -1)
    activity_cost = max(100.0, min(6000.0, activity_cost))

    recommended_duration = int(np.random.choice([2, 3, 4, 5, 6, 7, 8, 10], p=[0.15, 0.30, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]))
    popularity_score = round(float(np.random.uniform(3.5, 4.95)), 2)

    img_url = IMAGE_MAP.get(base_dest, IMAGE_MAP.get(base_state, "https://images.unsplash.com/photo-1477587458883-47145ed94245?auto=format&fit=crop&w=800&q=80"))

    records.append({
        'destination': dest_name,
        'state': base_state,
        'destination_type': base_type,
        'climate': base_climate,
        'best_season': base_season,
        'average_hotel_cost': hotel_cost,
        'average_food_cost': food_cost,
        'average_local_transport_cost': transport_cost,
        'average_activity_cost': activity_cost,
        'recommended_duration': recommended_duration,
        'popularity_score': popularity_score,
        'image_url': img_url
    })

df_synthetic = pd.DataFrame(records)

# Intentional duplicates & NaNs for preprocessing demonstration
missing_mask_hotel = np.random.rand(len(df_synthetic)) < 0.03
missing_mask_food = np.random.rand(len(df_synthetic)) < 0.02
df_synthetic.loc[missing_mask_hotel, 'average_hotel_cost'] = np.nan
df_synthetic.loc[missing_mask_food, 'average_food_cost'] = np.nan

duplicates = df_synthetic.iloc[[10, 25, 42, 67, 99]].copy()
df_synthetic = pd.concat([df_synthetic, duplicates], ignore_index=True)

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, "indian_destinations_raw.csv")

df_synthetic.to_csv(output_path, index=False)
print(f"Generated {len(df_synthetic)} synthetic raw Indian destination records with travel image URLs in {output_path}")
