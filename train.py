# train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import TimeSeriesDataset
from preprocess import preprocess
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import time
import json
from tqdm import tqdm
import os

from lstm_model import LSTMModel
from transformer_model import TransformerModel


MODEL = "LSTM"
# MODEL = "TRANSFORMER"


def main():
    start_time = time.time()

    print("Loading data...")
    raw_data = pd.read_csv("final_data.csv")
    data, scaler = preprocess(raw_data)
    print(f"Data loaded with shape: {data.shape}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    lookback = 96
    print(f"Using lookback period: {lookback}")

    train, valid = 0.7, 0.1
    train_size, valid_size = int(train * len(data)), int(valid * len(data))

    train_data = data[:train_size]
    valid_data = data[train_size : train_size + valid_size]
    test_data = data[train_size + valid_size :]

    print("Splitting data into training and testing sets...")
    train_dataset = TimeSeriesDataset(train_data, lookback=lookback)
    valid_dataset = TimeSeriesDataset(valid_data, lookback=lookback)
    test_dataset = TimeSeriesDataset(test_data, lookback=lookback)
    print("TimeSeriesDataset created.")

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    num_features = output_size = len(data.columns) - 1

    print("Initializing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if MODEL == "LSTM":
        model = LSTMModel(
            num_features=num_features,
            hidden_size=256,
            num_layers=16,
            output_size=output_size,
        ).to(device)

    elif MODEL == "TRANSFORMER":
        model = TransformerModel(
            num_features=num_features,
            d_model=256,
            nhead=8,
            num_encoder_layers=16,
            dim_feedforward=256,
            output_size=output_size,
        ).to(device)

    else:
        raise ValueError("Invalid model type. Use 'LSTM' or 'TRANSFORMER'")

    print("Model initialized.")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 20
    print("Starting model training...")
    training_start_time = time.time()
    train_losses = []
    valid_losses = []

    base_path = f"results_{MODEL.lower()}"

    os.makedirs(f"{base_path}/model", exist_ok=True)
    os.makedirs(f"{base_path}/loss", exist_ok=True)

    # Initialize variables to track the best validation loss
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = f"{base_path}/model/best_model.pth"

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train()
        # Initialize tqdm loader with a variable to set postfix
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in loop:
            batch_x = batch[0].to(device)
            batch_y = batch[1].to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update the postfix with the current loss
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            loop = tqdm(valid_loader, desc="Validation", leave=False)
            for batch in loop:
                batch_x = batch[0].to(device)
                batch_y = batch[1].to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(valid_loader)
        valid_losses.append(avg_val_loss)

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f} seconds"
        )

        # Check if the current validation loss is the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(
                f"Best model updated at epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}"
            )

        # Plot losses and save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                },
                f"{base_path}/model/epoch_{epoch+1}.pth",
            )

            plt.figure()
            plt.plot(range(1, epoch + 2), train_losses, label="Training Loss")
            plt.plot(range(1, epoch + 2), valid_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Losses")
            plt.legend()
            plt.savefig(f"{base_path}/loss/epoch_{epoch+1}.png")
            plt.close()

    total_training_time = time.time() - training_start_time
    print(
        f"Model training completed. Total training time: {total_training_time:.2f} seconds"
    )

    print("Loading the best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    print("Making predictions on the test set...")
    prediction_start_time = time.time()
    model.eval()
    y_pred = []
    y_test = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred.append(outputs.cpu())
            y_test.append(batch_y)
    y_pred = torch.cat(y_pred, dim=0)
    y_test = torch.cat(y_test, dim=0)
    print(f"Predictions completed in {time.time() - prediction_start_time:.2f} seconds")

    # Evaluate the model
    print("Evaluating the model...")
    eval_start_time = time.time()
    y_test_np = y_test.numpy()
    y_pred_np = y_pred.numpy()
    rmse = np.sqrt(mean_squared_error(y_test_np, y_pred_np))
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")

    print(f"y_test: {y_test_np.shape}")
    print(f"y_pred: {y_pred_np.shape}")

    # Inverse transform the predictions and actual values
    print("Inverse transforming the predictions and actual values...")
    transform_start_time = time.time()
    # Since scaler expects 2D input, we need to reshape
    y_test_inverse = scaler.inverse_transform(y_test_np)
    y_pred_inverse = scaler.inverse_transform(y_pred_np)
    print(
        f"Inverse transformation completed in {time.time() - transform_start_time:.2f} seconds"
    )

    # Plot feature predictions
    print("Generating plots...")
    plot_start_time = time.time()
    features = raw_data.columns[1:]
    num_plots = 8
    start_feat_num = 8
    for feat_num in range(start_feat_num, start_feat_num + num_plots):
        feat_name = features[feat_num]
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inverse[:, feat_num], label="Actual")
        plt.plot(y_pred_inverse[:, feat_num], label="Predicted")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(f"Actual vs Predicted Values for Feature {feat_name}")
        plt.legend()
        plt.savefig(
            f"results/{MODEL.lower()}/actual_vs_predicted_{feat_num - start_feat_num}.png"
        )
        plt.close()
    print(f"Plots generated in {time.time() - plot_start_time:.2f} seconds")

    print("Saving the model and training info ...")
    save_start_time = time.time()

    # torch.save(model.state_dict(), f"{base_path}/model/best_model_final.pth")

    model_info = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_features": num_features,
        "hidden_size": model.hidden_size if MODEL == "LSTM" else None,
        "num_layers": model.num_layers if MODEL == "LSTM" else None,
        "output_size": output_size,
        "lookback": lookback,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }
    model_info_path = f"{base_path}/{MODEL.lower()}_model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f)
    print(
        f"Model and training info saved in {time.time() - save_start_time:.2f} seconds"
    )

    total_time = time.time() - start_time
    print(
        f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
