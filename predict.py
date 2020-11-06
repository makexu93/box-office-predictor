import json
import os
import pickle as pkl

import numpy as np

import sagemaker_xgboost_container.encoder as xgb_encoders



def model_fn(model_dir):
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    return booster

from io import BytesIO
import numpy as np
import xgboost as xgb


def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array where the first columns are predictions
    and the remaining columns are the feature contributions (SHAP values) for that prediction.
    """
    prediction = model.predict(input_data)
    feature_contribs = model.predict(input_data, pred_contribs=True)
    output = np.hstack((prediction[:, np.newaxis], feature_contribs))
    return output

