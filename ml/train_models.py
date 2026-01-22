import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



# Load dataset
df = pd.read_csv("data/powergrid_dataset.csv")

# Feature columns
categorical_cols = [
    "project_type",
    "terrain_type",
    "permit_complexity",
    "material_availability",
    "demand_pressure"
]

numerical_cols = [
    "terrain_difficulty_score",
    "permit_delay_months",
    "vendor_rating",
    "skilled_manpower_ratio",
    "start_month",
    "weather_severity_index",
    "planned_cost_lakhs",
    "planned_duration_months"
]

X = df[categorical_cols + numerical_cols]
y_cost = df["actual_cost_lakhs"]
y_time = df["actual_duration_months"]

# Encode categorical data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_cat = encoder.fit_transform(df[categorical_cols])
encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(categorical_cols)
)

X_final = pd.concat([encoded_cat_df, df[numerical_cols]], axis=1)

# Train models
cost_model = RandomForestRegressor(n_estimators=200, random_state=42)
time_model = RandomForestRegressor(n_estimators=200, random_state=42)

cost_model.fit(X_final, y_cost)
time_model.fit(X_final, y_time)

# Save models and metadata
pickle.dump(cost_model, open("model/cost_model.pkl", "wb"))
pickle.dump(time_model, open("model/duration_model.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))
pickle.dump(X_final.columns.tolist(), open("model/feature_columns.pkl", "wb"))

print("✅ Models trained and saved successfully")
