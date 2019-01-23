from pytest import approx
from sklearn.impute import SimpleImputer 
import pandas as pd
import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

def test_imputer():
    pip1 = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ])
    x = np.array([1,2,3,np.nan]).reshape(-1,1)
    o1 = pip1.fit_transform(x)
    o1 = np.squeeze(o1,axis=1).tolist()
    ans = [1,2,3,2]
    assert o1 == ans


def test_std_scaler():
    pip1 = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
    ])
    x = np.array([1,2,3,np.nan]).reshape(-1,1)
    o1 = pip1.fit_transform(x)
    o1 = np.squeeze(o1,axis=1).tolist()
    mean = np.mean([1,2,3,2])
    std = np.std([1,2,3,2])
    ans = [1,2,3,2]
    ans = (ans - mean)/std
    assert o1 == approx(ans)