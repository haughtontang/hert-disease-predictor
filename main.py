import numpy as np
from predictor_src.data_loading import load_raw_data, scale_data, return_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # Load and transform the data
    raw_df = load_raw_data()
    # Remove rows where there are any nan values - I may want to do this more inteligently but I will remove it for now just to get it done
    raw_df = raw_df.dropna(how="any")
    features_to_scale = ["age", "trestbps", "chol", "thalach"]
    scaled_df = scale_data(data_df=raw_df, features_to_scale=features_to_scale)
    x_train, x_test = return_train_test_split(data_df=scaled_df)
    y_train, y_test = np.array(x_train["y"]), np.array(x_test["y"])
    # remove labels from training data
    del x_train["y"]
    del x_test["y"]

    # Create the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print("train accuracy", accuracy * 100)
    test_prediction = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, test_prediction)
    print("test accuracy", test_accuracy * 100)


if __name__ == '__main__':
    main()
