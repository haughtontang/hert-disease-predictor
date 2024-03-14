import typing

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# The place I got the data from says that a random forrest classification/classic logistic regression is the most accurate model to use


def load_raw_data():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    all_data_df = heart_disease.data.features
    targets_df = heart_disease.data.targets
    all_data_df["y"] = targets_df["num"]


def scale_data(data_df: pd.DataFrame, features_to_scale: list) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_columns = scaler.fit_transform(data_df[features_to_scale])
    data_df[features_to_scale] = scaled_columns
    return data_df


def return_train_test_split(data_df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_test = train_test_split(data_df, test_size=0.2)
    return x_train, x_test
