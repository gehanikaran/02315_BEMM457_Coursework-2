import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'vgsales.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Step 1: Handle Missing Values
# Remove rows with missing values in critical columns
df_cleaned = df.dropna(subset=['Year', 'Publisher'])

# Step 2: Validate Year Column
# Keep records within a valid range (e.g., 1980-2023)
df_cleaned = df_cleaned[(df_cleaned['Year'] >= 1980) & (df_cleaned['Year'] <= 2023)]

# Step 3: Remove Duplicates
# Drop duplicate rows if any
df_cleaned = df_cleaned.drop_duplicates()

# Step 4: Check and Fix Unrealistic Sales Data
# Ensure that Global Sales are greater than or equal to the sum of regional sales
df_cleaned = df_cleaned[
    df_cleaned['Global_Sales'] >= (
        df_cleaned['NA_Sales'] + df_cleaned['EU_Sales'] + df_cleaned['JP_Sales'] + df_cleaned['Other_Sales']
    )
]

# Step 5: Detect and Handle Outliers in Global Sales
# Use IQR to identify and remove extreme outliers
q1 = df_cleaned['Global_Sales'].quantile(0.25)
q3 = df_cleaned['Global_Sales'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_cleaned = df_cleaned[(df_cleaned['Global_Sales'] >= lower_bound) & (df_cleaned['Global_Sales'] <= upper_bound)]

# Step 6: Group Rare Categories in Platform and Genre
# Group platforms and genres with less than 10 occurrences into 'Other'
platform_counts = df_cleaned['Platform'].value_counts()
rare_platforms = platform_counts[platform_counts < 10].index
df_cleaned['Platform'] = df_cleaned['Platform'].replace(rare_platforms, 'Other')

genre_counts = df_cleaned['Genre'].value_counts()
rare_genres = genre_counts[genre_counts < 10].index
df_cleaned['Genre'] = df_cleaned['Genre'].replace(rare_genres, 'Other')

# Step 7: Encode Categorical Variables
# One-Hot Encoding for categorical columns (Platform, Genre)
categorical_columns = ['Platform', 'Genre']
encoder = OneHotEncoder(sparse_output=False)  # Corrected parameter
encoded_columns = encoder.fit_transform(df_cleaned[categorical_columns])

# Create a DataFrame with encoded columns
encoded_df = pd.DataFrame(
    encoded_columns,
    columns=encoder.get_feature_names_out(categorical_columns)
)

# Add encoded columns back into the dataset and remove the original text columns
df_cleaned = df_cleaned.reset_index(drop=True)  # Reset index for merging
df_encoded = pd.concat([df_cleaned, encoded_df], axis=1)
df_encoded.drop(categorical_columns, axis=1, inplace=True)

# Step 8: Scale Numerical Columns
# Standardize sales columns for better analysis
numerical_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(df_encoded[numerical_columns])


# Step 9: Round Sales Columns
# Round sales columns to 2 decimal places for consistency
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
df_cleaned[sales_columns] = df_cleaned[sales_columns].round(2)

# Create a DataFrame with scaled columns
scaled_df = pd.DataFrame(
    scaled_columns,
    columns=[f"scaled_{col}" for col in numerical_columns]
)

# Add scaled columns back into the dataset and remove the original sales columns
df_encoded = pd.concat([df_encoded, scaled_df], axis=1)
df_encoded.drop(numerical_columns, axis=1, inplace=True)

# Save the cleaned dataset
df_encoded.to_csv('vgsales_cleaned.csv', index=False)
print("\nCleaned dataset saved as 'vgsales_cleaned.csv'.")

# Summary of Cleaning
print("\nFinal Cleaned Dataset Overview:")
print(df_encoded.head())
