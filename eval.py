
import argparse
import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import TimeSeriesDataset
from preprocess import preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lstm_model import LSTMModel
from transformer_model import TransformerModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set and generate prediction plots."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["LSTM", "TRANSFORMER"],
        required=True,
        help="Type of the model to evaluate: 'LSTM' or 'TRANSFORMER'.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth file) to load.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="final_data.csv",
        help="Path to the CSV data file.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results and plots.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=96,
        help="Lookback period used for the TimeSeriesDataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_plots",
        type=int,
        default=8,
        help="Number of feature plots to generate.",
    )
    parser.add_argument(
        "--start_feat_num",
        type=int,
        default=8,
        help="Starting feature index for plotting.",
    )
    return parser.parse_args()


def load_model(
    model_type,
    num_features,
    hidden_size,
    num_layers,
    output_size,
    checkpoint_path,
    device,
):
    if model_type == "LSTM":
        model = LSTMModel(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
        )
    elif model_type == "TRANSFORMER":
        model = TransformerModel(
            num_features=num_features,
            d_model=256,
            nhead=8,
            num_encoder_layers=num_layers,
            dim_feedforward=256,
            output_size=output_size,
        )
    else:
        raise ValueError("Invalid model type. Use 'LSTM' or 'TRANSFORMER'.")

    model_state = torch.load(checkpoint_path, map_location=device)
    if "model" in model_state:
        model_state = model_state["model"]
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    start_time = time.time()

    os.makedirs(args.results_dir, exist_ok=True)
    plots_dir = os.path.join("plots", args.model_type.lower())
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading data...")
    raw_data = pd.read_csv(args.data_path)
    data, scaler = preprocess(raw_data)
    print(f"Data loaded with shape: {data.shape}")
    print(f"Time taken to load data: {time.time() - start_time:.2f} seconds")

    lookback = args.lookback
    print(f"Using lookback period: {lookback}")

    train_ratio, valid_ratio = 0.7, 0.1
    train_size = int(train_ratio * len(data))
    valid_size = int(valid_ratio * len(data))

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]
    test_data = data[train_size + valid_size :]

    print(len(test_data))

    print("Creating TimeSeriesDataset for test data...")
    test_dataset = TimeSeriesDataset(test_data, lookback=lookback)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Test DataLoader created.")

    num_features = len(data.columns) - 1  # Assuming the last column is the target
    output_size = num_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = os.path.dirname(args.checkpoint)
    model_info_path = os.path.join(checkpoint_dir, "model_info.json")
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        hidden_size = model_info.get("hidden_size", 256)
        num_layers = model_info.get("num_layers", 2)
    else:
        hidden_size = 256
        num_layers = 8
        print("model_info.json not found. Using default model parameters.")

    print("Loading the model from checkpoint...")
    model = load_model(
        model_type=args.model_type,
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    print("Model loaded successfully.")

    print("Making predictions on the test set...")
    prediction_start_time = time.time()
    y_pred = []
    y_test = []
    with torch.no_grad():
        i = 0
        for batch in tqdm(test_loader, desc="Predicting"):
            i += 1
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)
            outputs = model(batch_x)
            y_pred.append(outputs.cpu())
            y_test.append(batch_y.cpu())

    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_test = torch.cat(y_test, dim=0).numpy()
    print(f"Predictions completed in {time.time() - prediction_start_time:.2f} seconds")

    print("Evaluating the model...")
    eval_start_time = time.time()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred shape: {y_pred.shape}")

    print("Inverse transforming the predictions and actual values...")
    transform_start_time = time.time()
    y_test_inverse = scaler.inverse_transform(y_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)
    print(
        f"Inverse transformation completed in {time.time() - transform_start_time:.2f} seconds"
    )

    print("Generating prediction plots...")
    plot_start_time = time.time()
    features = raw_data.columns[1:]

    stations = set(["_".join(i.split("_")[:2]) for i in features if "site" in i])
    scores = {i: {} for i in stations}

    pollutants = [
        "pm25",
        "pm10",
        "ozone",
        "no2",
        "so2",
        "co",
        "nox",
        "no",
        "nh3",
    ]

    for i, feat in enumerate(features):
        s = None
        for _s in stations:
            if _s in feat:
                s = _s
                break

        if s is None:
            continue

        pollutant = None
        for p in pollutants:
            if feat.endswith(p):
                pollutant = p
                break

        if pollutant is None:
            continue

        mse = mean_squared_error(y_test_inverse[:, i], y_pred_inverse[:, i])
        mape = mean_absolute_percentage_error(
            y_test_inverse[:, i], y_pred_inverse[:, i]
        )
        scores[s][pollutant] = {"mse": mse, "mape": mape, "rmse": np.sqrt(mse)}

    for s in scores:
        mse = sum([scores[s][p]["mse"] for p in scores[s]]) / len(scores[s])
        mape = sum([scores[s][p]["mape"] for p in scores[s]]) / len(scores[s])
        scores[s]["avg"] = {"mse": mse, "mape": mape, "rmse": np.sqrt(mse)}

    # Convert this into a df with the columns: station (Get name instead of site, ex: site_103_crri_mathura_road_delhi_co), pollutant, mse, mape
    df = []
    for s in scores:
        for p in scores[s]:
            df.append(
                {
                    "station": s,
                    "pollutant": p,
                    "mse": scores[s][p]["mse"],
                    "mape": scores[s][p]["mape"],
                    "rmse": scores[s][p]["rmse"],
                }
            )

    df = pd.DataFrame(df)
    print(df.sort_values(by=["mape"]))
    print(df.sort_values(by=["mse"]))
    df.to_csv(f"results_{args.model_type.lower()}.csv")

    for feat_num in range(len(features)):
        feat_name = features[feat_num]

        if not any(i in feat_name for i in ["site_122", "site_5393"]):
            continue

        print(f"Plotting feature '{feat_name}'...")

        start = 21000
        print(len(y_test_inverse))

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inverse[start : start + 1000, feat_num], label="Actual")
        plt.plot(y_pred_inverse[start : start + 1000, feat_num], label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(f"Actual vs Predicted Values for Feature '{feat_name}'")
        plt.legend()
        plot_path = os.path.join(plots_dir, f"{feat_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    print(f"Plots generated in {time.time() - plot_start_time:.2f} seconds")

    total_time = time.time() - start_time
    print(
        f"\nTotal evaluation time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
