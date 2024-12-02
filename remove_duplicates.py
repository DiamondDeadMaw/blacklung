import pandas as pd
from functools import reduce

csv_files = ['cbh.csv', 'cvh.csv', 'e.csv', 'sp.csv', 't2m.csv',
             'tcc.csv', 'tp.csv', 'u10.csv', 'v10.csv']

csv_files_left = ['cbh.csv', 'cvh.csv']
dataframes = []
path = "./data/"
c = 0
variable_name = "cbh"
time_column = "valid_time"

df = pd.read_csv(path + f"{variable_name}.csv")
pd.set_option('display.max_columns', None)
df = df[[time_column, "latitude", "longitude", variable_name]]

new_df = pd.DataFrame(columns=[time_column, "latitude", "longitude", variable_name])
print(df.head())
total = len(df)
print(f"Total rows: {total}")

startIndex = 0
dupeSize = 0
for i in range(len(df)):
    if i % 10000 == 0:
        print(i*100/total)

    row = df.iloc[i]
    startRow = df.iloc[startIndex]

    if row[time_column] == startRow[time_column]:
        dupeSize += 1
    else:
        sum_u10 = sum(df.iloc[startIndex:i][variable_name])
        mean_u10 = sum_u10 / dupeSize

        avg_row = {
            f"{time_column}": startRow[time_column],
            "latitude": startRow["latitude"],
            "longitude": startRow["longitude"],
            f"{variable_name}": mean_u10
        }

        new_df = pd.concat([new_df, pd.DataFrame([avg_row])], ignore_index=True)
        startIndex = i
        dupeSize = 1
sum_u10 = sum(df.iloc[startIndex:][variable_name])
mean_u10 = sum_u10 / dupeSize
avg_row = {
    f"{time_column}": df.iloc[startIndex][time_column],
    "latitude": df.iloc[startIndex]["latitude"],
    "longitude": df.iloc[startIndex]["longitude"],
    f"{variable_name}": mean_u10
}
new_df = pd.concat([new_df, pd.DataFrame([avg_row])], ignore_index=True)

new_df.to_csv(f"merged_data_{variable_name}.csv", index=False)
