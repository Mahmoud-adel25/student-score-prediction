import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # type: ignore
import pandas as pd

def train_and_evaluate(df, features, target, model_type="Linear Regression"):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Polynomial Regression (degree=2)":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
        return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), y_test, y_pred, model, poly  
    else:  # Linear Regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), y_test, y_pred, model, None

def get_enhanced_metrics(y_true, y_pred):
    """Get comprehensive model evaluation metrics"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate adjusted R²
    n = len(y_true)
    p = len(y_true)  # number of features (approximate)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    return {
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

def get_feature_importance(model, feature_names, model_type):
    """Get feature importance for different model types"""
    if model_type in ["Random Forest", "Gradient Boosting"]:
        return dict(zip(feature_names, model.feature_importances_))
    elif model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression"]:
        return dict(zip(feature_names, np.abs(model.coef_)))
    else:
        return None

import joblib

def train_and_save_model(df, features, target, model_type="Linear"):
    X = df[features]
    y = df[target]

    if model_type == "Polynomial":
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LinearRegression
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("lin", LinearRegression())
        ])
        transformer_type = "Polynomial(degree=2)"
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        transformer_type = "None"

    model.fit(X, y)

    # Save model
    joblib.dump(model, "saved_model.pkl")

    # Save feature columns used in training
    joblib.dump(X.columns.tolist(), "trained_model_features.pkl")

    # Save meta-info
    meta = {"model_type": model_type, "transformer_type": transformer_type}
    joblib.dump(meta, "model_metadata.pkl")

def get_model_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual Exam Score")
    ax.set_ylabel("Predicted Exam Score")
    ax.set_title("Actual vs Predicted")
    return fig  
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residuals")
    return fig  # ✅ return the figure

def predict_new_score(model, transformer, features, user_input):
    input_df = pd.DataFrame([user_input])[features]

    if transformer is not None:
        input_df = transformer.transform(input_df)
    print("✅ Input to Model:\n", input_df)
    prediction = model.predict(input_df)[0]
    print("🎯 Prediction:", prediction)
    return prediction

def predict_with_preprocessing(model, transformer, features, user_input, original_df):
    """
    Make prediction with proper preprocessing that matches training pipeline
    """
    from src.data_cleaning import clean_data, get_fitted_scaler
    
    # Get fitted scaler from original training data
    fitted_scaler = get_fitted_scaler(original_df)
    
    # Create input DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Apply same cleaning as training (with fitted scaler)
    input_df_cleaned = clean_data(input_df, use_onehot=True, normalize=True, remove_outliers=False, scaler=fitted_scaler)
    
    # Ensure all model features are present
    for feature in features:
        if feature not in input_df_cleaned.columns:
            input_df_cleaned[feature] = 0.0
    
    # Select only features used in training
    input_df_final = input_df_cleaned[features]
    
    # Apply transformer if polynomial regression
    if transformer is not None:
        input_transformed = transformer.transform(input_df_final)
    else:
        input_transformed = input_df_final
    
    # Make prediction
    prediction = model.predict(input_transformed)[0]
    
    return prediction, input_df_final, input_transformed
