#load

import pandas as pd 

df = pd.read_csv("Car_data.csv")

print(df.head(10))

print("Shape:", df.shape)

print("Columns:", df.columns.tolist())

print(df.info())

print(df.describe())


#Explainatory data analysis

import matplotlib.pyplot as plt

print("missing values:\n", df.isnull().sum())

print("Duplicate rows:", df.duplicated().sum())

print(df.describe())

plt.figure(figsize=(8,5))
plt.hist(df['Selling_Price'], bins=20, edgecolor='black')
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price (Lakhs)')
plt.ylabel('Frequency')
plt.show()

numeric_cols =['Selling_Price', 'Present_Price', 'Kms_Driven']
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

    plt.figure(figsize=(8,5))
plt.scatter(df['Present_Price'], df['Selling_Price'], alpha=0.6)
plt.title('Relationship between Present Price and Selling Price')
plt.xlabel('Present Price (Lakhs)')
plt.ylabel('Selling Price (Lakhs)')
plt.show()

numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()

plt.figure(figsize=(6,5))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

print(corr)