from sklearn.linear_model import LassoCV
import numpy as np

def get_features_lasso(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    lasso = LassoCV(cv=5).fit(X, y)
    coef = lasso.coef_

    selected = X.columns[np.abs(coef) > 1e-5]
    return list(selected)
