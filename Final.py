import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('FinalData10Stations.csv')
print("Read data")
aqi_pollutants = ["pm25", "pm10", "ozone", "no2", "so2", "co"]

def split_columns(data_columns):
    targets = [col for col in data_columns if any(pollutant in col.lower() for pollutant in aqi_pollutants)]
    features = [col for col in data_columns if col not in targets]
    return targets, features

print("Splitting columns")
data_columns = data.columns
targets, features = split_columns(data_columns)


# timestamp column is always in index 0, remove it
del features[0]

print("Creating extra features")

print("lagged")
# Create lagged features
for target in targets:
    data[f'{target}_t-1'] = data[target].shift(1)
    data[f'{target}_t-7'] = data[target].shift(7)
    data[f'{target}_t-30'] = data[target].shift(30)
    data[f'{target}_t-365'] = data[target].shift(365)
print("window")
# Rolling window features
window_sizes = [7, 14, 30]
for target in targets:
    for window in window_sizes:
        data[f'{target}_rolling_mean_{window}'] = data[target].rolling(window).mean()
        data[f'{target}_rolling_std_{window}'] = data[target].rolling(window).std()
print("Time based")
# Time-based features
data['day_of_week'] = data['Timestamp'].apply(lambda x: pd.to_datetime(x).dayofweek)
data['month'] = data['Timestamp'].apply(lambda x: pd.to_datetime(x).month)

# Combine all features
X = data[features + [f'{target}_t-1' for target in targets] + [f'{target}_t-7' for target in targets] +
          [f'{target}_t-30' for target in targets] + [f'{target}_t-365' for target in targets] +
          [f'{target}_rolling_mean_{window}' for window in window_sizes for target in targets] +
          [f'{target}_rolling_std_{window}' for window in window_sizes for target in targets] +
          ['day_of_week', 'month']]
y = data[targets]

print("Imputation")
# Imputation
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
y_imputed = imputer.fit_transform(y)

print("Detect outliers")
outlier_detector = IsolationForest(contamination=0.05, random_state=42)
outliers = outlier_detector.fit_predict(X_imputed)
X_imputed = X_imputed[outliers == 1]
y_imputed = y_imputed[outliers == 1]

print("Feature scaling")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

def mean_absolute_percentage_error(y_true, y_pred):
    epsilon = 1e-10
    y_true = np.where(y_true == 0, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Time based split
train_size = int(len(X_scaled) * 0.8)
X_train_full = X_scaled[:train_size]
X_test = X_scaled[train_size:]
y_train_full = y_imputed[:train_size]
y_test = y_imputed[train_size:]

# Split again into percentages
val_size = int(len(X_train_full) * 0.2)
X_train = X_train_full[:-val_size]
X_val = X_train_full[-val_size:]
y_train = y_train_full[:-val_size]
y_val = y_train_full[-val_size:]

model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.6,
    colsample_bytree=0.8,
    missing=np.nan,
    random_state=42,
    tree_method="hist",
    device="cuda"
)

model_filename = "xgb_model.json"

print("Training")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True,
)

print("Evaluating on validation set")
y_val_pred = model.predict(X_val)

for i, target in enumerate(targets):
    mape = mean_absolute_percentage_error(y_val[:, i], y_val_pred[:, i])
    r2 = r2_score(y_val[:, i], y_val_pred[:, i])
    print(f"Target: {target}")
    print(f"  Validation MAPE: {mape:.4f}%")
    print(f"  Validation R² Score: {r2:.4f}")

print("Selecting features")
selector = SelectFromModel(model, threshold="mean", max_features=30)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

print("Retraining model with selected features")
model.fit(
    X_train_selected, y_train,
    eval_set=[(X_val_selected, y_val)],
    verbose=True,
)
model.save_model(model_filename)

print("Evaluating on test set")
y_test_pred = model.predict(X_test_selected)

for i, target in enumerate(targets):
    mape = mean_absolute_percentage_error(y_test[:, i], y_test_pred[:, i])
    r2 = r2_score(y_test[:, i], y_test_pred[:, i])
    print(f"Target: {target}")
    print(f"  Test MAPE: {mape:.4f}%")
    print(f"  Test R² Score: {r2:.4f}")

def plot_actual_vs_predicted(y_actual, y_predicted, targets, num_features_to_plot=5):
    chosen_features = np.random.choice(range(len(targets)), num_features_to_plot, replace=False)

    fig, axes = plt.subplots(nrows=num_features_to_plot, figsize=(12, num_features_to_plot * 4))
    for i, feature_idx in enumerate(chosen_features):
        axes[i].plot(y_actual[:, feature_idx], label="Actual", alpha=0.7)
        axes[i].plot(y_predicted[:, feature_idx], label="Predicted", alpha=0.7)
        axes[i].set_title(f"Actual vs Predicted for Target: {targets[feature_idx]}")
        axes[i].set_xlabel("Sample Index")
        axes[i].set_ylabel("Value")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(y_test, y_test_pred, targets)

from sklearn.metrics import mean_squared_error

# CPCB standards aqi calc
def calculate_aqi(pollutant_concentrations):
    pollutant_aqi_values = []

    for pollutant, concentration in pollutant_concentrations.items():
        if np.isnan(concentration):
            continue
        breakpoints = get_breakpoints_for_pollutant(pollutant)
        aqi = pollutant_aqi(concentration, breakpoints)
        pollutant_aqi_values.append(aqi)

    if pollutant_aqi_values:
        return max(pollutant_aqi_values)
    else:
        return None

def get_breakpoints_for_pollutant(pollutant):
    if pollutant == 'pm25':
        return [
            {"lower": 0, "upper": 30, "index_lower": 0, "index_upper": 50},
            {"lower": 31, "upper": 60, "index_lower": 51, "index_upper": 100},
            {"lower": 61, "upper": 90, "index_lower": 101, "index_upper": 200},
            {"lower": 91, "upper": 120, "index_lower": 201, "index_upper": 300},
            {"lower": 121, "upper": 250, "index_lower": 301, "index_upper": 400},
        ]
    elif pollutant == 'pm10':
        return [
            {"lower": 0, "upper": 50, "index_lower": 0, "index_upper": 50},
            {"lower": 51, "upper": 100, "index_lower": 51, "index_upper": 100},
            {"lower": 101, "upper": 250, "index_lower": 101, "index_upper": 200},
            {"lower": 251, "upper": 350, "index_lower": 201, "index_upper": 300},
            {"lower": 351, "upper": 430, "index_lower": 301, "index_upper": 400},
        ]
    elif pollutant == 'ozone':
        return [
            {"lower": 0, "upper": 50, "index_lower": 0, "index_upper": 50},
            {"lower": 51, "upper": 100, "index_lower": 51, "index_upper": 100},
            {"lower": 101, "upper": 168, "index_lower": 101, "index_upper": 200},
            {"lower": 169, "upper": 208, "index_lower": 201, "index_upper": 300},
            {"lower": 209, "upper": 748, "index_lower": 301, "index_upper": 400},
        ]
    elif pollutant == 'no2':
        return [
            {"lower": 0, "upper": 40, "index_lower": 0, "index_upper": 50},
            {"lower": 41, "upper": 80, "index_lower": 51, "index_upper": 100},
            {"lower": 81, "upper": 180, "index_lower": 101, "index_upper": 200},
            {"lower": 181, "upper": 280, "index_lower": 201, "index_upper": 300},
            {"lower": 281, "upper": 400, "index_lower": 301, "index_upper": 400},
        ]
    elif pollutant == 'so2':
        return [
            {"lower": 0, "upper": 40, "index_lower": 0, "index_upper": 50},
            {"lower": 41, "upper": 80, "index_lower": 51, "index_upper": 100},
            {"lower": 81, "upper": 380, "index_lower": 101, "index_upper": 200},
            {"lower": 381, "upper": 800, "index_lower": 201, "index_upper": 300},
            {"lower": 801, "upper": 1600, "index_lower": 301, "index_upper": 400},
        ]
    elif pollutant == 'co':
        return [
            {"lower": 0, "upper": 1, "index_lower": 0, "index_upper": 50},
            {"lower": 1.1, "upper": 2, "index_lower": 51, "index_upper": 100},
            {"lower": 2.1, "upper": 10, "index_lower": 101, "index_upper": 200},
            {"lower": 10.1, "upper": 17, "index_lower": 201, "index_upper": 300},
            {"lower": 17.1, "upper": 34, "index_lower": 301, "index_upper": 400},
        ]
    else:
        return []

def pollutant_aqi(concentration, breakpoints):
    for bp in breakpoints:
        if concentration <= bp["upper"]:
            return ((bp["index_upper"] - bp["index_lower"]) /
                    (bp["upper"] - bp["lower"])) * (concentration - bp["lower"]) + bp["index_lower"]
    return 500  # Max AQI if above the highest breakpoint

stations = ['site_103_crri_mathura_road_delhi', 'site_106_igi_airport_(t3)_delhi', 'site_108_aya_nagar_delhi', 'site_114_ihbas_dilshad_garden_delhi', 'site_105_north_campus_du_delhi_solar', 'site_104_burari_crossing_delhi', 'site_109_lodhi_road_delhi', 'site_113_shadipur_delhi', 'site_115_nsit_dwarka_delhi', 'site_107_pusa_delhi', 'site_105_north_campus_du_delhi']
pollutants = ["pm25", "pm10", "ozone", "no2", "so2", "co"]
station_pollutant_indices = {}

# Get columns to look at
for station in stations:
    pollutant_indices = {}
    for pollutant in pollutants:
        col_name = f"{station}_{pollutant}"
        print(f"Col name: {col_name}")
        if col_name in targets:
            idx = targets.index(col_name)
            pollutant_indices[pollutant] = idx
    if pollutant_indices:
        station_pollutant_indices[station] = pollutant_indices
    else:
        print(f"No pollutants found for station {station}.")

actual_aqi = []
predicted_aqi = []

for station, pollutant_indices in station_pollutant_indices.items():
    for i in range(len(y_test)):
        try:
            actual_values = {}
            predicted_values = {}
            for pollutant, idx in pollutant_indices.items():
                actual_values[pollutant] = y_test[i, idx]
                predicted_values[pollutant] = y_test_pred[i, idx]

            actual_aqi_value = calculate_aqi(actual_values)
            predicted_aqi_value = calculate_aqi(predicted_values)

            if actual_aqi_value is not None and predicted_aqi_value is not None:
                actual_aqi.append(actual_aqi_value)
                predicted_aqi.append(predicted_aqi_value)
        except IndexError as e:
            print(f"Index error for station {station} at sample {i}: {e}")
            continue

if actual_aqi and predicted_aqi:
    mse = mean_squared_error(actual_aqi, predicted_aqi)
    print(f"Mean Squared Error (MSE) for AQI: {mse:.4f}")
else:
    print("No AQI values to compute MSE.")


if actual_aqi and predicted_aqi:
    plt.figure(figsize=(12, 6))
    plt.plot(actual_aqi, label="Actual AQI", alpha=0.7)
    plt.plot(predicted_aqi, label="Predicted AQI", alpha=0.7)
    plt.title("Actual vs Predicted AQI")
    plt.xlabel("Sample Index")
    plt.ylabel("AQI")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("No AQI values to plot.")