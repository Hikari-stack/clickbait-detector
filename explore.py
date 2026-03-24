import pandas as pd

# Load the dataset
df = pd.read_csv('archive/clickbait_data.csv')

# Show basic info
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:", df.columns.tolist())

print("\nClass distribution:")
print(df.iloc[:, 1].value_counts())

print("\nAny missing values?")
print(df.isnull().sum())
