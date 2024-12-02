import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.impute import SimpleImputer  # Simple mean imputation
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class AQIDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32)
        }
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4, num_layers=2, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, 32)
        self.transformer = nn.Transformer(
            d_model=32,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=64,
            dropout=dropout
        )
        self.fc_out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        transformer_out = self.transformer(x, x)
        transformer_out = transformer_out.mean(dim=1)
        output = self.fc_out(transformer_out)
        return output

def preprocess_data_with_imputation(df, pollutant_columns, weather_columns, lookback_period=96):
    for col in pollutant_columns:
        df[f'{col}_lag'] = df[col].shift(lookback_period)

    lagged_columns = [f'{col}_lag' for col in pollutant_columns]
    features = df[lagged_columns + weather_columns].values
    targets = df[pollutant_columns].values

    # Imputation using mean
    imputer = SimpleImputer(strategy='mean')
    features = imputer.fit_transform(features)
    targets = imputer.fit_transform(targets)

    # Normalize features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features)
    targets = target_scaler.fit_transform(targets)

    return features, targets, feature_scaler, target_scaler

df = pd.read_csv("FinalData10Stations.csv")

pollutant_columns = [col for col in df.columns if 'pm' in col or 'so2' in col or 'no' in col or 'ozone' in col or 'co' in col]
weather_columns = [col for col in df.columns if col not in pollutant_columns and col != 'Timestamp']

features, targets, feature_scaler, target_scaler = preprocess_data_with_imputation(df, pollutant_columns, weather_columns)

train_size = int(0.8 * len(features))
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = targets[:train_size], targets[train_size:]

train_dataset = AQIDataset(X_train, y_train)
test_dataset = AQIDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = X_train.shape[1]
output_dim = len(pollutant_columns)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TransformerModel(input_dim=input_dim, output_dim=output_dim).to(device)

# Loss and optimizer
criterion = nn.L1Loss()  # MAE Loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        with tqdm(total=len(train_loader), desc="Training") as pbar:
            for batch in train_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)

        train_loss /= len(train_loader)
        print(f"Training Loss: {train_loss:.4f}")
        scheduler.step()

        # Save checkpoint
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")


def calculate_aqi(pollutant_values):
    aqi_breakpoints = {
        'pm25': [
            (0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ],
        'pm10': [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500)
        ],
        'co': [
            (0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500)
        ],
        'no2': [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 150, 101, 150),
            (151, 200, 151, 200),
            (201, 300, 201, 300),
            (301, 400, 301, 400),
            (401, 500, 401, 500)
        ],
        'nox': [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 150, 101, 150),
            (151, 200, 151, 200),
            (201, 300, 201, 300),
            (301, 400, 301, 400),
            (401, 500, 401, 500)
        ],
        'ozone': [
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 300, 301, 400),
            (301, 500, 401, 500)
        ],
        'so2': [
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500)
        ]
    }

    def get_pollutant_aqi(value, pollutant):
        # pollutant type (last part of column name)
        pollutant_type = pollutant.split('_')[-1]

        # Find appropriate breakpoints
        for bp in aqi_breakpoints.get(pollutant_type, []):
            if bp[0] <= value <= bp[1]:
                # Linear interpolation
                bp_low, bp_high, aqi_low, aqi_high = bp
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (value - bp_low) + aqi_low
                return aqi

        return 500 if value > 500 else 0

    # Calculate AQI for each pollutant
    pollutant_aqis = {}
    for pollutant, value in pollutant_values.items():
        pollutant_aqis[pollutant] = get_pollutant_aqi(value, pollutant)

    # Return max AQI
    return max(pollutant_aqis.values())

def evaluate_model(model, test_loader, criterion, device, target_scaler, pollutant_columns):
    model.eval()
    test_loss = 0
    smape = 0
    mape = 0
    print("Evaluating model on test data...")
    all_predictions = []
    all_targets = []
    with tqdm(total=len(test_loader), desc="Evaluating") as pbar:
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # Compute SMAPE
                smape += torch.mean(
                    2 * torch.abs(targets - outputs) / (torch.abs(targets) + torch.abs(outputs) + 1e-8)
                ) * 100
                # Compute MAPE
                mape += torch.mean(torch.abs((targets - outputs) / (torch.abs(targets) + 1e-8))) * 100
                pbar.update(1)

    test_loss /= len(test_loader)
    smape /= len(test_loader)
    mape /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test SMAPE: {smape:.2f}%")
    print(f"Test MAPE: {mape:.2f}%")

    # Convert predictions and targets to numpy arrays for plotting
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Plot Actual vs Predicted for randomly chosen features
    num_features_to_plot = 5
    chosen_features = np.random.choice(range(len(pollutant_columns)), num_features_to_plot, replace=False)

    for feature_idx in chosen_features:
        plt.figure(figsize=(10, 6))
        plt.plot(all_targets[:, feature_idx], label="Actual", alpha=0.7)
        plt.plot(all_predictions[:, feature_idx], label="Predicted", alpha=0.7)
        plt.title(f"Actual vs Predicted for Feature {pollutant_columns[feature_idx]}")
        plt.xlabel("Sample Index")
        plt.ylabel("Value (scaled)")
        plt.legend()
        plt.grid(True)
        plt.show()


def evaluate_model_with_aqi(model, test_loader, criterion, device, target_scaler, pollutant_columns,
                                          original_df):
    model.eval()
    print("Evaluating model with  AQI calculation...")

    all_predictions = []
    all_targets = []
    all_aqi_predictions = []
    all_aqi_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(features)

            # Inverse transform to get original scale
            outputs_original = target_scaler.inverse_transform(outputs.cpu().numpy())
            targets_original = target_scaler.inverse_transform(targets.cpu().numpy())

            all_predictions.append(outputs_original)
            all_targets.append(targets_original)

            # Calculate  AQI for each sample
            batch_aqi_predictions = []
            batch_aqi_targets = []

            for sample_pred, sample_target in zip(outputs_original, targets_original):
                pred_dict = {pollutant: value for pollutant, value in zip(pollutant_columns, sample_pred)}
                target_dict = {pollutant: value for pollutant, value in zip(pollutant_columns, sample_target)}

                pred_aqi = calculate_aqi(pred_dict)
                target_aqi = calculate_aqi(target_dict)

                batch_aqi_predictions.append(pred_aqi)
                batch_aqi_targets.append(target_aqi)

            all_aqi_predictions.append(batch_aqi_predictions)
            all_aqi_targets.append(batch_aqi_targets)

    all_aqi_predictions = np.concatenate(all_aqi_predictions)
    all_aqi_targets = np.concatenate(all_aqi_targets)

    _aqi_mse = mean_squared_error(all_aqi_targets, all_aqi_predictions)
    print(f" AQI Mean Squared Error: {_aqi_mse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(all_aqi_targets, label="Actual  AQI", alpha=0.7)
    plt.plot(all_aqi_predictions, label="Predicted  AQI", alpha=0.7)
    plt.title(" AQI: Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel(" AQI")
    plt.legend()
    plt.grid(True)
    plt.show()

    return _aqi_mse, all_aqi_predictions, all_aqi_targets



# Load checkpoint weights, if not training
model = TransformerModel(input_dim=X_train.shape[1], output_dim=len(pollutant_columns)).to(device)
checkpoint_path = "checkpoint_epoch_50.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Loaded weights from {checkpoint_path}")
_aqi_mse, predicted_aqi, target_aqi = evaluate_model_with_aqi(
    model,
    test_loader,
    criterion,
    device,
    target_scaler,
    pollutant_columns,
    df
)
print(_aqi_mse)


def categorize_aqi(aqi_values):
    categories = []
    for aqi in aqi_values:
        if aqi <= 50:
            categories.append("Good")
        elif 51 <= aqi <= 100:
            categories.append("Moderate")
        elif 101 <= aqi <= 150:
            categories.append("Unhealthy for Sensitive Groups")
        elif 151 <= aqi <= 200:
            categories.append("Unhealthy")
        elif 201 <= aqi <= 300:
            categories.append("Very Unhealthy")
        else:
            categories.append("Hazardous")
    labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
    for l in labels:
        if l not in categories:
            categories.append(l)
    return categories


# Updated evaluation with confusion matrix
def evaluate_model_with_confusion_matrix(model, test_loader, criterion, device, target_scaler, pollutant_columns):
    _aqi_mse, predicted_aqi, target_aqi = evaluate_model_with_aqi(
        model, test_loader, criterion, device, target_scaler, pollutant_columns, df
    )

    # Categorize AQIs
    predicted_aqi_categories = categorize_aqi(predicted_aqi)
    target_aqi_categories = categorize_aqi(target_aqi)

    # Confusion Matrix generation
    labels = ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]
    cm = confusion_matrix(target_aqi_categories, predicted_aqi_categories, labels=labels)

    print("\nConfusion Matrix:")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for AQI Categories")
    plt.show()

    print("\nClassification Report:")
    print(classification_report(target_aqi_categories, predicted_aqi_categories, target_names=labels))

    return _aqi_mse, cm, predicted_aqi, target_aqi


_aqi_mse, cm, predicted_aqi, target_aqi = evaluate_model_with_confusion_matrix(
    model, test_loader, criterion, device, target_scaler, pollutant_columns
)
