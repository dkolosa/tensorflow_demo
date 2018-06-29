import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # load traning_test dataset
    training_data_df = pd.read_csv("sales_data_training.csv", dtype=float)

    x_training = training_data_df.drop('total_earnings', axis=1).values
    y_training = training_data_df[['total_earnings']].values

    test_data_df = pd.read_csv("sales_data_test.csv", dtype=float)
    x_test = test_data_df.drop('total_earnings', axis=1).values
    y_test = test_data_df[['total_earnings']].values

    x_scaler = MinMaxScaler(feature_range=(0,1))
    y_scaler = MinMaxScaler(feature_range=(0,1))

    x_scaled_training = x_scaler.fit_transform(x_training)
    y_scaled_training = y_scaler.fit_transform(y_training)

    x_scaled_testing = x_scaler.transform(x_test)
    y_scaled_testing = y_scaler.transform(y_test)

    print("Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(y_scaler.scale_[0], y_scaler.min_[0]))

    return [x_scaled_training, y_scaled_training, x_scaled_testing, y_scaled_testing]

