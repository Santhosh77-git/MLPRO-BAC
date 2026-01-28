import pickle
import pandas as pd
import shap
from flask import Flask, request, jsonify

app = Flask(__name__)

# =====================================================
# LOAD MODELS & METADATA (LOAD ONCE → FAST)
# =====================================================
with open("model/cost_model.pkl", "rb") as f:
    cost_model = pickle.load(f)

with open("model/duration_model.pkl", "rb") as f:
    duration_model = pickle.load(f)

with open("model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("model/feature_columns.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

CATEGORICAL_COLS = encoder.feature_names_in_.tolist()

# =====================================================
# SHAP EXPLAINERS (TREE → LIGHTWEIGHT)
# =====================================================
cost_explainer = shap.TreeExplainer(cost_model)
duration_explainer = shap.TreeExplainer(duration_model)

# =====================================================
# HUMAN LABELS & ACTIONS
# =====================================================
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

# =====================================================
# HELPER: PREPROCESS INPUT
# =====================================================
def preprocess_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    # Ensure all categorical columns exist
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "unknown"

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str).fillna("unknown")
    numerical_cols = [c for c in df.columns if c not in CATEGORICAL_COLS]

    # Encode categorical
    encoded = encoder.transform(df[CATEGORICAL_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLS)
    )

    final_df = pd.concat([encoded_df, df[numerical_cols]], axis=1)
    final_df = final_df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return final_df

# =====================================================
# HELPER: SHAP ANALYSIS
# =====================================================
def compute_shap(final_df: pd.DataFrame):
    shap_cost = cost_explainer.shap_values(final_df)[0]
    shap_duration = duration_explainer.shap_values(final_df)[0]

    shap_total = pd.Series(
        shap_cost + shap_duration,
        index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    return shap_total

# =====================================================
# HELPER: WHAT-IF ANALYSIS
# =====================================================
def what_if_analysis(user_input, factor, values):
    results = []

    for v in values:
        modified = user_input.copy()
        modified[factor] = v

        temp_df = preprocess_input(modified)
        pred_cost = round(float(cost_model.predict(temp_df)[0]), 2)

        results.append(pred_cost)

    return results

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json

    # -------------------------
    # PREPROCESS & PREDICT
    # -------------------------
    final_df = preprocess_input(user_input)

    cost = round(float(cost_model.predict(final_df)[0]), 2)
    duration = round(float(duration_model.predict(final_df)[0]), 2)

    # -------------------------
    # SHAP EXPLANATION
    # -------------------------
    shap_total = compute_shap(final_df)

    risks, actions = [], []

    for feature, impact in shap_total.items():
        if impact > 0:
            for key in FEATURE_LABELS:
                if key in feature:
                    risks.append(FEATURE_LABELS[key])
                    actions.append(RECOMMENDATIONS[key])

    risks = list(dict.fromkeys(risks))[:3]
    actions = list(dict.fromkeys(actions))[:3]

    # -------------------------
    # GRAPH DATA
    # -------------------------

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

    # 3️⃣ What-If (Top 3 Factors)
    what_if_graphs = []

    WHAT_IF_FACTORS = {
        "vendor_rating": [1, 2, 3, 4, 5],
        "weather_severity_index": [0, 1, 2, 3, 4],
        "skilled_manpower_ratio": [0.4, 0.6, 0.8, 1.0, 1.2]
    }

    for factor, values in WHAT_IF_FACTORS.items():
        y_vals = what_if_analysis(user_input, factor, values)

        what_if_graphs.append({
            "factor": factor.replace("_", " ").title(),
            "x": values,
            "y": y_vals,
            "current_value": user_input.get(factor)
        })

    # -------------------------
    # RESPONSE
    # -------------------------
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
            "what_if": what_if_graphs
        }
    })

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
