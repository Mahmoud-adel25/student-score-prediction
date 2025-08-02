
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_data(df, use_onehot=True, normalize=True, remove_outliers=True, scaler=None):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Fill missing categorical values
    fill_cols = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Encode categorical features
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if use_onehot:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    else:
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])

    # Remove outliers (only for training data)
    if remove_outliers:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Normalize
    if normalize:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        if "Exam_Score" in num_cols:
            num_cols = num_cols.drop("Exam_Score")
        
        if scaler is None:
            # Create new scaler for training
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        else:
            # Use existing scaler for prediction
            df[num_cols] = scaler.transform(df[num_cols])

    return df

def get_fitted_scaler(df, use_onehot=True):
    """Get a fitted scaler from training data"""
    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.strip()

    # Fill missing categorical values
    fill_cols = ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]
    for col in fill_cols:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].fillna("Unknown")

    # Encode categorical features
    cat_cols = df_temp.select_dtypes(include="object").columns.tolist()
    if use_onehot:
        df_temp = pd.get_dummies(df_temp, columns=cat_cols, drop_first=True)
    
    # Get numerical columns for scaling
    num_cols = df_temp.select_dtypes(include=["int64", "float64"]).columns
    if "Exam_Score" in num_cols:
        num_cols = num_cols.drop("Exam_Score")
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(df_temp[num_cols])
    
    return scaler
