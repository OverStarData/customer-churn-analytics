import os
import io
import base64
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

# Load preprocessor + model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/Gradient_Boosting.joblib")

# Raw input columns (19 fields)
input_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
              'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
              'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input
        input_data = {col: request.form[col] for col in input_cols}

        # Convert numerics
        for col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            input_data[col] = float(input_data[col])

        df = pd.DataFrame([input_data])

        # Derive tenure_bin
        def tenure_bin_func(tenure):
            if tenure <= 12:
                return "0-1yr"
            elif tenure <= 24:
                return "1-2yr"
            elif tenure <= 48:
                return "2-4yr"
            elif tenure <= 60:
                return "4-5yr"
            else:
                return "5+yr"

        df["tenure_bin"] = df["tenure"].apply(tenure_bin_func)

        # Transform using saved preprocessor
        X = preprocessor.transform(df)

        # Predict
        prob = model.predict_proba(X)[0, 1]
        pred = "Churn" if prob >= 0.5 else "Not Churn"

        return render_template("predict.html",
                               prediction=pred,
                               probability=round(prob, 3))
    except Exception as e:
        return render_template("index.html", error=str(e))



# expect app.static_folder to be configured, e.g. 'static'

@app.route("/dataset")
def dataset():
    DATA_PATH = os.path.join(app.root_path, "data", "processed", "telco_churn_step1_clean.csv")
    df = pd.read_csv(DATA_PATH)

    # Summary
    n_rows, n_cols = df.shape
    n_missing = 0
    n_duplicates = int(df.duplicated().sum())
    churn_counts = df["Churn"].value_counts().to_dict()
    churn_rate = round(churn_counts.get("Yes", 0) / max(n_rows, 1) * 100, 2)

    # Profiling
    profiling = []
    for col in df.columns:
        if df[col].dtype == "object":
            top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "-"
            profiling.append({
                "column": col, "dtype": "Categorical",
                "unique": int(df[col].nunique()),
                "missing": "00",
                "top_value": top_val, "mean": "-", "std": "-"
            })
        else:
            profiling.append({
                "column": col, "dtype": "Numeric",
                "unique": int(df[col].nunique()),
                "missing": 0,
                "top_value": "-", "mean": round(df[col].mean(), 3),
                "std": round(df[col].std(), 3)
            })

    # Correlation heatmap → save to static/generated
    static_dir = os.path.join(app.root_path, "static", "generated")
    os.makedirs(static_dir, exist_ok=True)
    corr_img_path = os.path.join(static_dir, "corr_heatmap.png")

    try:
        corr = df.corr(numeric_only=True).round(2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.tight_layout()
        plt.savefig(corr_img_path, dpi=150)
        plt.close()
    except Exception as e:
        print("Correlation heatmap failed:", e)
        corr_img_path = None

    # Feature importance
    feature_importance = []
    try:
        gb_model = joblib.load(os.path.join(app.root_path, "models", "Gradient_Boosting.joblib"))
        if hasattr(gb_model, "feature_importances_"):
            fi = gb_model.feature_importances_
            try:
                preprocessor = joblib.load(os.path.join(app.root_path, "models", "preprocessor.joblib"))
                feature_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feature_names = [f"feat_{i}" for i in range(len(fi))]

            fi_list = sorted(
                [{"feature": f, "importance": float(round(im, 6))}
                 for f, im in zip(feature_names, fi)],
                key=lambda x: x["importance"], reverse=True
            )
            feature_importance = fi_list[:20]
    except Exception as e:
        feature_importance = [{"feature": "Error", "importance": str(e)}]

    # Sample (send as JSON instead of HTML)
    sample_df = df.sample(min(100, len(df)))
    sample_data = sample_df.reset_index().to_dict(orient="records")  # keeps original index
    sample_columns = ["index"] + df.columns.tolist()  # put index first

    # Tableau images
    tableau_dir = os.path.join(app.root_path, "static", "tableau")
    tableau_images = []
    if os.path.exists(tableau_dir):
        for fname in sorted(os.listdir(tableau_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                tableau_images.append(url_for("static", filename=f"tableau/{fname}"))

    # Numeric distributions
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    distributions = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            distributions.append({
                "column": col,
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max())
            })

    return render_template("dataset.html",
                           n_rows=n_rows, n_cols=n_cols,
                           n_missing=n_missing, n_duplicates=n_duplicates,
                           profiling=profiling, churn_counts=churn_counts,
                           churn_rate=churn_rate,
                           correlation_image=url_for("static", filename="generated/corr_heatmap.png") if corr_img_path else None,
                           feature_importance=feature_importance,
                           sample_columns=sample_columns,
                           sample_data=sample_data,
                           distributions=distributions,
                           tableau_images=tableau_images)


if __name__ == "__main__":
    app.run(debug=True)
