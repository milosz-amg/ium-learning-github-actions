from sklearn.model_selection import train_test_split
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os

# Folder z plikami CSV
DATA_FOLDER = "data"

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def normalize(df):
    # Copy the original DataFrame
    df_processed = df.copy()

    # Drop 'Unnamed: 0' if exists
    df_processed = df_processed.drop(columns=["Unnamed: 0"], errors="ignore")
    
    # Define numeric and non-numeric columns
    columns_numeric = df_processed.select_dtypes(include=['number']).columns
    columns_non_numeric = df_processed.select_dtypes(exclude=['number']).columns

    # Convert categorical columns to lowercase
    for col in columns_non_numeric:
        df_processed[col] = df_processed[col].map(lambda x: x.lower() if isinstance(x, str) else x)

    # One-hot encoding for categorical columns
    df_processed = pd.get_dummies(df_processed, columns=columns_non_numeric, drop_first=True)

    # Normalize numeric columns to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_processed[columns_numeric] = scaler.fit_transform(df_processed[columns_numeric])

    # Round numeric columns to 2 decimal places
    df_processed[columns_numeric] = df_processed[columns_numeric].round(2)

    return df_processed

def remove_artifacts(df):
    df_cleaned = df.copy()
    
    # Liczba wierszy przed czyszczeniem
    initial_rows = df_cleaned.shape[0]

    # Usunięcie całkowicie pustych wierszy
    df_cleaned.dropna(how="all", inplace=True)

    # Usunięcie wierszy z wartościami znacznie odchylonymi od średniej (>4σ)
    columns_numeric = df_cleaned.select_dtypes(include=['number']).columns
    for col in columns_numeric:
        mean = df_cleaned[col].mean()
        std = df_cleaned[col].std()
        df_cleaned = df_cleaned[(df_cleaned[col] >= mean - 4 * std) & (df_cleaned[col] <= mean + 4 * std)]

    # Liczba usuniętych wierszy
    removed_rows = initial_rows - df_cleaned.shape[0]

    return df_cleaned, removed_rows

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=7)
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=7)
    return train_df, test_df, valid_df


# Utworzenie folderu na dane
create_folder(DATA_FOLDER)

# Obsługa argumentu CUTOFF
try:
    CUTOFF = int(sys.argv[1])
    print(f"CUTOFF set to: {CUTOFF}")
except (IndexError, ValueError):
    CUTOFF = 0  # Domyślnie 0 = nie ograniczaj
    print("No valid CUTOFF provided. Using full dataset.")

# Load the diamonds dataset
ds = load_dataset("jdxcosta/diamonds")
df = ds["train"].to_pandas() 

# Trim the dataset if CUTOFF > 0
if CUTOFF > 0:
    df = df.head(CUTOFF)  # Można zamienić na sample jeśli chcesz losowo
    print(f"Dataset trimmed to {len(df)} rows.")

# Data preprocessing
df = normalize(df)
df, removed_rows = remove_artifacts(df)
print(f"Removed rows: {removed_rows}")

# Split the dataset
df_train, df_test, df_valid = split_data(df)

# Save the datasets to CSV files
df_train.to_csv("data/diamonds_train.csv", index=False)
df_test.to_csv("data/diamonds_test.csv", index=False)
df_valid.to_csv("data/diamonds_valid.csv", index=False)