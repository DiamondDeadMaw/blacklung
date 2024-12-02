# preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def preprocess(data):
    # Assume the first column is 'timestamp' and exclude it from scaling
    timestamp_column = 'Timestamp'
    feature_columns = [col for col in data.columns if col != timestamp_column]

    # Separate features and timestamp
    timestamps = data[[timestamp_column]]
    features = data[feature_columns]

    # Identify columns with all NaN values
    all_nan_columns = features.columns[features.isna().all()]
    if len(all_nan_columns) > 0:
        print(f"Dropping columns with all NaN values: {list(all_nan_columns)}")
        features = features.drop(columns=all_nan_columns)
        feature_columns = features.columns  # Update feature_columns

    # Initialize the MinMaxScaler with the desired feature range
    scaler = MinMaxScaler(feature_range=(0, 1))

    features_filled = features.ffill().bfill()

    scaled_features = scaler.fit_transform(features_filled)

    # Create a DataFrame for the scaled features
    scaled_features_df = pd.DataFrame(scaled_features, columns=feature_columns)

    # Restore original NaN positions
    nan_mask = features.isna()
    scaled_features_df = scaled_features_df.mask(nan_mask)
    scaled_features_df = scaled_features_df.ffill().bfill()

    # Plot a sample of scaled vs original data to verify scaling
    plt.figure(figsize=(12, 6))
    plt.plot(features.iloc[:, 8], label='Original', alpha=0.7)
    plt.plot(scaled_features_df.iloc[:, 8], label='Scaled', alpha=0.7)
    plt.title('Sample Feature: Original vs Scaled Values')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('scaling_verification.png')
    plt.close()

    # Combine the timestamp column with the scaled features
    processed_data = pd.concat([timestamps, scaled_features_df], axis=1)
    return processed_data, scaler


def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file, nrows=10000)

    processed_data, scaler = preprocess(data)

    processed_data.to_csv(output_file, index=False)
    print(f"Data has been preprocessed and saved to '{output_file}'.")
    print("A verification plot has been saved as 'scaling_verification.png'")


if __name__ == "__main__":
    input_file = "final_postcleaning.csv"
    output_file = "FINAL_PREPROCESSED.csv"
    preprocess_data(input_file, output_file)
