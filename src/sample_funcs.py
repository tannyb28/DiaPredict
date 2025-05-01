import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_xy(df: pd.DataFrame, target: str='Diabetes_binary'):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

def make_train_test(df: pd.DataFrame, **kwargs):
    X, y = split_xy(df)
    return train_test_split(X, y, **kwargs)

def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
      ('scale', StandardScaler()),
      ('clf',  LogisticRegression(max_iter=1000))
    ])

def fit_pipeline(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline