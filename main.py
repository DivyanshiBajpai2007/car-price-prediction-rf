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



#Feature Engineering

df = df.drop_duplicates()

Q1 = df['Kms_Driven'].quantile(0.25)
Q3 = df['Kms_Driven'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

print("Upper limit for Kms_Driven:", upper_limit)
print("Rows above limit:", (df['Kms_Driven'] > upper_limit).sum())

df['Kms_Driven'] = df['Kms_Driven'].clip(upper=upper_limit)


df = df.reset_index(drop=True)

df['Car_Age'] = 2025 - df['Year']


df['Mileage_per_Year'] = df['Kms_Driven'] / (df['Car_Age'] + 1)


premium_keywords = [
    'innova', 'fortuner', 'corolla', 'camry', 'land cruiser', 'etios',  # Toyota
    'city', 'brio', 'amaze', 'jazz', 'cb ', 'cbr', 'karizma', 'activa', 'cb-',  # Honda
    'i20', 'i10', 'eon', 'xcent', 'verna', 'creta', 'elantra'  # Hyundai
]

df['Premium_Brand'] = df['Car_Name'].str.lower().apply(
    lambda name: 1 if any(keyword in name for keyword in premium_keywords) else 0
)


median_kms = df['Kms_Driven'].median()
df['High_Mileage'] = (df['Kms_Driven'] > median_kms).astype(int)


print(df[['Car_Name', 'Car_Age', 'Mileage_per_Year', 'Premium_Brand', 'High_Mileage']].head(10))