import pandas as pd
import numpy as np
import pytest

from src.sample_funcs import load_data, split_xy, make_train_test, build_logistic_pipeline, fit_pipeline

@pytest.fixture
def sample_df(tmp_path):
    #Dataframe with snippet of sample diabetes dataset
    df = pd.DataFrame({
        'Feat1': [0,1,0,1],
        'Feat2': [1.0, 1.0, 1.0, 1.0],
        'Diabetes_binary': [0,1,0,1]
    })
    f = tmp_path / "mini.csv"
    df.to_csv(f, index=False)
    return f
#Test loading, separating features/targets, and train/test split
def test_load_and_split(sample_df):
    df = load_data(str(sample_df))
    assert 'Diabetes_binary' in df.columns
    X, y = split_xy(df)
    # make sure drop/selection worked
    assert list(X.columns) == ['Feat1', 'Feat2']
    assert np.issubdtype(y.dtype, np.integer)

#Test split logic for validity
def test_train_test_shapes(sample_df):
    df = load_data(str(sample_df))
    X_train, X_test, y_train, y_test = make_train_test(df, test_size=0.5, random_state=0)
    assert X_train.shape[0] == 2
    assert X_test.shape[0]  == 2
    assert len(y_train) == 2
    assert len(y_test)  == 2

#Test scaler and LogReg pipeline fitting and prediction
def test_pipeline_fits(sample_df):
    df = load_data(str(sample_df))
    X_train, X_test, y_train, y_test = make_train_test(df, test_size=0.5, random_state=0)
    pipe = build_logistic_pipeline()
    fitted = fit_pipeline(pipe, X_train, y_train)
    preds = fitted.predict(X_test)
    assert len(preds) == X_test.shape[0]