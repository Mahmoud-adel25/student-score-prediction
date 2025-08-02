import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st # type: ignore
import pandas as pd
from sklearn.linear_model import LinearRegression
def plot_eda(df):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    clear_df = df.select_dtypes(include=["int64", "float64"])
    sns.heatmap(clear_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution of Exam Score")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(y=df["Exam_Score"], ax=ax1, color="skyblue")
    ax1.set_title("Boxplot of Exam Score")

    sns.histplot(df["Exam_Score"], kde=True, ax=ax2, color="lightgreen")
    ax2.set_title("Histogram of Exam Score")

    st.pyplot(fig)

def plot_top_correlations(df, target_col, top_n=10):
    numeric_corr = df.corr(numeric_only=True)
    if target_col not in numeric_corr.columns:
        st.warning(f"{target_col} not found in correlation matrix.")
        return

    sorted_corr = numeric_corr[target_col].drop(target_col).abs().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sorted_corr.values, y=sorted_corr.index, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Correlated Features with {target_col}")
    ax.set_xlabel("Correlation (absolute)")
    st.pyplot(fig)

def plot_feature_importance(df, features, target):
    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    importance = pd.Series(model.coef_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind='barh', ax=ax, color="steelblue")
    ax.set_title("Linear Model Coefficients (Feature Importance)")
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig)