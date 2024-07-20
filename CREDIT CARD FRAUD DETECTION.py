import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# Load the datasets
df_train = pd.read_csv("fraudTrain.csv")
df_test = pd.read_csv("fraudTest.csv")

# Concatenate the train and test datasets
df_combined = pd.concat([df_train, df_test])

# Display basic information about the datasets
print(f'Train shape: {df_train.shape}')
print(f'Test shape: {df_test.shape}')
print(df_combined.head())
print(df_combined.describe())
print(df_combined.isnull().sum())

# Label encode categorical columns
label_encoders = {}
categorical_columns = ['merchant', 'category', 'gender', 'state', 'job']

for col in categorical_columns:
    encoder = LabelEncoder()
    df_combined[col] = encoder.fit_transform(df_combined[col])
    label_encoders[col] = encoder

# Convert datetime columns to separate year, month, day, and hour columns
for df in [df_combined, df_train, df_test]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['transaction_year'] = df['trans_date_trans_time'].dt.year
    df['transaction_month'] = df['trans_date_trans_time'].dt.month
    df['transaction_day'] = df['trans_date_trans_time'].dt.day
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour

    df['birth_year'] = df['dob'].dt.year
    df['birth_month'] = df['dob'].dt.month
    df['birth_day'] = df['dob'].dt.day

    df.drop(['trans_date_trans_time', 'dob'], axis=1, inplace=True)

# Drop unnecessary columns
columns_to_drop = ['first', 'last', 'street', 'city', 'trans_num']
df_combined.drop(columns_to_drop, axis=1, inplace=True)
df_train.drop(columns_to_drop, axis=1, inplace=True)
df_test.drop(columns_to_drop, axis=1, inplace=True)

print(f'Train shape after preprocessing: {df_train.shape}')
print(f'Test shape after preprocessing: {df_test.shape}')
print(f'Combined data shape after preprocessing: {df_combined.shape}')

print(df_combined.head())
print(df_combined.describe())
print(df_combined.isnull().sum())

# Plot the distribution of fraudulent vs non-fraudulent transactions
sns.countplot(data=df_combined, x='is_fraud')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
plt.show()

# Check for and reset duplicate indices
print(f'Duplicate indices: {df_combined.index.duplicated().sum()}')
df_combined = df_combined.reset_index(drop=True)
print(f'Duplicate indices after reset: {df_combined.index.duplicated().sum()}')

# Plot transaction counts by category and fraud status
plt.figure(figsize=(12, 6))
sns.countplot(data=df_combined, y='category', hue='is_fraud')
plt.title('Transaction Counts by Category and Fraud Status')
plt.show()

# Plot transaction counts by gender and fraud status
sns.countplot(data=df_combined, x='gender', hue='is_fraud')
plt.title('Transaction Counts by Gender and Fraud Status')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(df_combined.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

# Split the data into features and target variable
X_features = df_combined.drop('is_fraud', axis=1)
y_target = df_combined['is_fraud']

# Split the data into training and testing sets
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_features, y_target, test_size=0.2, random_state=42)

# Train and evaluate Logistic Regression model
model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train_split, y_train_split)
y_pred_logistic = model_logistic.predict(X_test_split)
print("Logistic Regression")
print(classification_report(y_test_split, y_pred_logistic))
print(confusion_matrix(y_test_split, y_pred_logistic))
print("Accuracy:", accuracy_score(y_test_split, y_pred_logistic))

# Train and evaluate Decision Tree model
model_decision_tree = DecisionTreeClassifier()
model_decision_tree.fit(X_train_split, y_train_split)
y_pred_tree = model_decision_tree.predict(X_test_split)
print("Decision Tree")
print(classification_report(y_test_split, y_pred_tree))
print(confusion_matrix(y_test_split, y_pred_tree))
print("Accuracy:", accuracy_score(y_test_split, y_pred_tree))

# Train and evaluate Random Forest model
model_random_forest = RandomForestClassifier()
model_random_forest.fit(X_train_split, y_train_split)
y_pred_rf = model_random_forest.predict(X_test_split)
print("Random Forest")
print(classification_report(y_test_split, y_pred_rf))
print(confusion_matrix(y_test_split, y_pred_rf))
print("Accuracy:", accuracy_score(y_test_split, y_pred_rf))
