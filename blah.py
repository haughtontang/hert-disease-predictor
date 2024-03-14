from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# The place I got the data from says that a random forrest classification/classic logistic regression is the most accurate model to use

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

features_to_scale = ["age", "trestbps", "chol", "thalach"]
scaler = StandardScaler()
all_data_df = heart_disease.data.features
scaled_columns = scaler.fit_transform(all_data_df[features_to_scale])
all_data_df[features_to_scale] = scaled_columns
targets_df = heart_disease.data.targets
all_data_df["y"] = targets_df["num"]
x_train, x_test = train_test_split(all_data_df, test_size=0.2)
y_train, y_test = np.array(x_train["y"]), np.array(x_test["y"])
# remove labels from training data
del x_train["y"]
del x_test["y"]
# metadata
print(heart_disease.metadata)

# variable information
print(heart_disease.variables)
