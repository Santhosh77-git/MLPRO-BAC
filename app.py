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

    # ======================
    # PREPROCESSING
    # ======================
    categorical_cols = encoder.feature_names_in_.tolist()

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "unknown"

    df[categorical_cols] = df[categorical_cols].astype(str).fillna("unknown")
    numerical_cols = [col for col in df.columns if col not in categorical_cols]

    encoded_cat = encoder.transform(df[categorical_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    num_df = df[numerical_cols].reset_index(drop=True)
    final_df = pd.concat([encoded_cat_df, num_df], axis=1)
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    # ======================
    # PREDICTIONS
    # ======================
    cost = round(float(cost_model.predict(final_df)[0]), 2)
    duration = round(float(duration_model.predict(final_df)[0]), 2)

    # ======================
    # SHAP
    # ======================
    shap_cost = cost_explainer.shap_values(final_df)[0]
    shap_duration = duration_explainer.shap_values(final_df)[0]

    shap_total = pd.Series(
        shap_cost + shap_duration,
        index=feature_columns
    ).sort_values(ascending=False)

    risks, actions = [], []

    for feature, impact in shap_total.items():
        if impact > 0:
            for key in FEATURE_LABELS:
                if key in feature:
                    risks.append(FEATURE_LABELS[key])
                    actions.append(RECOMMENDATIONS[key])

    risks = list(dict.fromkeys(risks))[:3]
    actions = list(dict.fromkeys(actions))[:3]

    # ======================
    # 📊 GRAPH DATA
    # ======================

    # 1️⃣ Planned vs Predicted
    planned_vs_predicted = {
        "cost": {
            "labels": ["Planned", "Predicted"],
            "values": [
                user_input.get("planned_cost_lakhs", 0),
                cost
            ]
        },
        "duration": {
            "labels": ["Planned", "Predicted"],
            "values": [
                user_input.get("planned_duration_months", 0),
                duration
            ]
        }
    }

    # 2️⃣ Top Factors Pie
    factor_scores = {}
    for feature, value in shap_total.items():
        for key in FEATURE_LABELS:
            if key in feature:
                factor_scores[key] = factor_scores.get(key, 0) + abs(value)

    top_factors = sorted(
        factor_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top_factors_graph = {
        "labels": [FEATURE_LABELS[k] for k, _ in top_factors],
        "values": [round(v * 100, 2) for _, v in top_factors]
    }

    # 3️⃣ What-If Analysis (Vendor Rating)
    what_if_x = [1, 2, 3, 4, 5]
    what_if_y = []

    for rating in what_if_x:
        modified = user_input.copy()
        modified["vendor_rating"] = rating

        temp_df = pd.DataFrame([modified])
        temp_df[categorical_cols] = temp_df[categorical_cols].astype(str)

        enc = encoder.transform(temp_df[categorical_cols])
        enc_df = pd.DataFrame(
            enc,
            columns=encoder.get_feature_names_out(categorical_cols)
        )

        temp_final = pd.concat(
            [enc_df, temp_df[numerical_cols]],
            axis=1
        ).reindex(columns=feature_columns, fill_value=0)

        what_if_y.append(
            round(float(cost_model.predict(temp_final)[0]), 2)
        )

    what_if_graph = {
        "factor": "Vendor Rating",
        "x": what_if_x,
        "y": what_if_y,
        "current_value": user_input.get("vendor_rating")
    }

    # ======================
    # RESPONSE
    # ======================
    return jsonify({
        "prediction": {
            "predicted_cost_lakhs": cost,
            "predicted_duration_months": duration
        },
        "explanation": {
            "key_risk_factors": risks,
            "recommendations": actions
        },
        "graphs": {
            "planned_vs_predicted": planned_vs_predicted,
            "top_factors": top_factors_graph,
            "what_if": what_if_graph
        }
    })



if __name__ == "__main__":
    app.run(debug=True)
