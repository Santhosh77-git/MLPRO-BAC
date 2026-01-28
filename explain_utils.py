import pandas as pd

def get_top_factors(shap_cost, shap_duration, feature_columns, labels, top_n=5):
    shap_total = (
        pd.Series(shap_cost + shap_duration, index=feature_columns)
        .sort_values(key=abs, ascending=False)
    )

    factor_scores = {}
    for feature, value in shap_total.items():
        for key in labels:
            if key in feature:
                factor_scores[key] = factor_scores.get(key, 0) + abs(value)

    top = sorted(
        factor_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top
