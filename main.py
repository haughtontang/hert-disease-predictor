import numpy as np
from predictor_src.data_loading import load_raw_data, scale_data, return_train_test_split


def main():
    raw_df = load_raw_data()
    features_to_scale = ["age", "trestbps", "chol", "thalach"]
    scaled_df = scale_data(data_df=raw_df, features_to_scale=features_to_scale)
    x_train, x_test = return_train_test_split(data_df=scaled_df)
    y_train, y_test = np.array(x_train["y"]), np.array(x_test["y"])
    # remove labels from training data
    del x_train["y"]
    del x_test["y"]
