import shap
import pandas as pd
import pickle

cost_model = pickle.load(open("model/cost_model.pkl", "rb"))

def explain_prediction(input_df):
    explainer = shap.TreeExplainer(cost_model)
    shap_values = explainer.shap_values(input_df)

    shap_df = pd.DataFrame(
        shap_values,
        columns=input_df.columns
    )

    impact = shap_df.iloc[0].sort_values(key=abs, ascending=False)
    return impact.head(5)
