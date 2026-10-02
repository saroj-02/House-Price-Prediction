import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/Housing.csv"
MODEL_PATH = "models/linear_regression_model.pkl"
PLOT_PATH = "outputs/predicted_vs_actual.png"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    print("\nDataset columns:")
    print(df.columns.tolist())

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing data...")

    # Target column in Housing.csv
    target = "price"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


# ============================================================
# BUILD PREPROCESSING PIPELINE
# ============================================================

def build_pipeline(X):

    # Numerical features
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Categorical features
    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # Numerical pipeline
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical pipeline
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # Complete ML pipeline
    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LinearRegression()
            )
        ]
    )

    return model_pipeline


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model_pipeline, X_train, y_train):

    print("\nTraining Linear Regression model...")

    model_pipeline.fit(
        X_train,
        y_train
    )

    print("Model training completed successfully.")

    return model_pipeline


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_pipeline,
    X_test,
    y_test
):

    print("\nEvaluating model...")

    predictions = model_pipeline.predict(X_test)

    # R²
    r2 = r2_score(
        y_test,
        predictions
    )

    # MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # RMSE
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:,.2f}")
    print(f"RMSE     : {rmse:,.2f}")

    print("=" * 50)

    return predictions, r2, mae, rmse


# ============================================================
# PLOT ACTUAL VS PREDICTED
# ============================================================

def create_prediction_plot(
    y_test,
    predictions
):

    print("\nCreating predicted vs actual graph...")

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.scatterplot(
        x=y_test,
        y=predictions,
        alpha=0.7
    )

    # Perfect prediction line
    min_value = min(
        y_test.min(),
        predictions.min()
    )

    max_value = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel(
        "Actual House Price"
    )

    plt.ylabel(
        "Predicted House Price"
    )

    plt.title(
        "Actual vs Predicted House Prices"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300
    )

    plt.show()

    print(
        f"Graph saved to: {PLOT_PATH}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model_pipeline):

    print("\nSaving model...")

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model_pipeline,
        MODEL_PATH
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
    print("=" * 60)

    # 1. Load dataset
    df = load_data()

    # 2. Prepare data
    X, y = prepare_data(df)

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # 4. Build pipeline
    model_pipeline = build_pipeline(X)

    # 5. Train model
    model_pipeline = train_model(
        model_pipeline,
        X_train,
        y_train
    )

    # 6. Evaluate model
    predictions, r2, mae, rmse = evaluate_model(
        model_pipeline,
        X_test,
        y_test
    )

    # 7. Create graph
    create_prediction_plot(
        y_test,
        predictions
    )

    # 8. Save model
    save_model(
        model_pipeline
    )

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()