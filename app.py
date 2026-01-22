import pickle
import pandas as pd
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

# ======================
# LOAD PICKLE FILES
# ======================
cost_model = pickle.load(open("model/cost_model.pkl", "rb"))
duration_model = pickle.load(open("model/duration_model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))
feature_columns = pickle.load(open("model/feature_columns.pkl", "rb"))

# ======================
# SHAP EXPLAINERS
# ======================
cost_explainer = shap.TreeExplainer(cost_model)
duration_explainer = shap.TreeExplainer(duration_model)

# ======================
# LABELS & RECOMMENDATIONS
# ======================
FEATURE_LABELS = {
    "terrain_type_hilly": "Hilly terrain",
    "material_availability_low": "Low material availability",
    "weather_severity_index": "Adverse weather conditions",
    "vendor_rating": "Low vendor performance",
    "permit_complexity": "High regulatory complexity",
    "skilled_manpower_ratio": "Insufficient skilled manpower"
}

RECOMMENDATIONS = {
    "terrain_type_hilly": "Deploy specialized machinery and experienced contractors",
    "material_availability_low": "Maintain buffer inventory and advance procurement",
    "weather_severity_index": "Reschedule activities outside adverse seasons",
    "vendor_rating": "Engage higher-rated vendors with strict SLAs",
    "permit_complexity": "Initiate parallel clearance processes",
    "skilled_manpower_ratio": "Increase skilled manpower availability"
}

# ======================
# PREDICTION ENDPOINT
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json
    df = pd.DataFrame([user_input])

    # 🔹 Must EXACTLY match training categorical columns
    categorical_cols = encoder.feature_names_in_.tolist()

    # 🔹 Ensure all categorical columns exist
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"

    # 🔹 Force categorical columns to string & handle NaN
    df[categorical_cols] = (
        df[categorical_cols]
        .astype(str)
        .fillna("unknown")
    )

    # 🔹 Numerical columns = everything else
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    # 🔹 Encode categorical features ONLY
    encoded_cat = encoder.transform(df[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # 🔹 Numerical dataframe
    num_df = df[numerical_cols].reset_index(drop=True)

    # 🔹 Combine encoded + numerical
    final_df = pd.concat([encoded_cat_df, num_df], axis=1)

    # 🔹 Align column order EXACTLY as training
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    # 🔹 Predictions
    cost = round(float(cost_model.predict(final_df)[0]), 2)
    duration = round(float(duration_model.predict(final_df)[0]), 2)

    # 🔹 SHAP explanations
    shap_cost = cost_explainer.shap_values(final_df)[0]
    shap_duration = duration_explainer.shap_values(final_df)[0]

    shap_total = pd.Series(
        shap_cost + shap_duration,
        index=feature_columns
    ).sort_values(ascending=False)

    risks = []
    actions = []

    for feature, impact in shap_total.items():
        if impact > 0:
            for key in FEATURE_LABELS:
                if key in feature:
                    risks.append(FEATURE_LABELS[key])
                    actions.append(RECOMMENDATIONS[key])

    # 🔹 Remove duplicates & limit output
    risks = list(dict.fromkeys(risks))[:3]
    actions = list(dict.fromkeys(actions))[:3]

    return jsonify({
        "predicted_cost_lakhs": cost,
        "predicted_duration_months": duration,
        "key_risk_factors": risks,
        "recommendations": actions
    })


if __name__ == "__main__":
    app.run(debug=True)
