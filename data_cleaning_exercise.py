# ==============================================
# Data Cleaning Exercise: Handling Missing Values
# ==============================================
# Author: (Takudzwa)
# Description:
# This script explores missing data handling using the San Francisco Building Permits dataset.
# It demonstrates identifying, dropping, and imputing missing values.
# ==============================================

import pandas as pd
import numpy as np

def main():
    # --- Load dataset ---
    print("Loading dataset...")
    sf_permits = pd.read_csv(
        "../input/building-permit-applications-data/Building_Permits.csv",
        low_memory=False
    )
    np.random.seed(0)
    print("Dataset loaded successfully.\n")

    # --- Preview data ---
    print("First 5 rows of the dataset:")
    print(sf_permits.head(), "\n")

    # --- Check for missing values ---
    print("Checking for missing values...")
    missing_values_table = sf_permits.isnull().sum()
    print("Missing values per column:")
    print(missing_values_table[missing_values_table > 0], "\n")

    # --- Calculate percentage of missing values ---
    sum_of_all = np.prod(sf_permits.shape)
    missing_values = sf_permits.isnull().sum().sum()
    percent_missing_val = (missing_values / sum_of_all) * 100
    print(f"Total missing values: {missing_values}")
    print(f"Percentage of missing data: {percent_missing_val:.2f}%\n")

    # --- Drop missing values (rows) ---
    print("Dropping rows with any missing values...")
    new_sf_permit = sf_permits.dropna()
    print(f"Remaining rows after drop: {new_sf_permit.shape[0]}\n")

    # --- Drop missing values (columns) ---
    print("Dropping columns with any missing values...")
    sf_permits_with_na_dropped = sf_permits.dropna(axis=1)
    dropped_columns = sf_permits.shape[1] - sf_permits_with_na_dropped.shape[1]
    print(f"Number of columns dropped: {dropped_columns}")
    print(f"Remaining columns: {sf_permits_with_na_dropped.shape[1]}\n")

    # --- Fill missing values (imputation) ---
    print("Imputing missing values (backfill, then fill remaining with 0)...")
    sf_permits_with_na_imputed = sf_permits.bfill(axis=0).fillna(0)
    print("Imputation complete.\n")

    # --- Summary ---
    print("Summary of operations:")
    print(f"→ Dataset shape: {sf_permits.shape}")
    print(f"→ Columns dropped: {dropped_columns}")
    print(f"→ Rows dropped (dropna): {sf_permits.shape[0] - new_sf_permit.shape[0]}")
    print(f"→ % missing data before cleaning: {percent_missing_val:.2f}%")
    print("\nData cleaning process complete!")

if __name__ == "__main__":
    main()
