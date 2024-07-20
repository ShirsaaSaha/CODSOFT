import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Load and explore the dataset
df_churn = pd.read_csv("Churn_Modelling.csv")
print(df_churn.head())
print(df_churn.info())

# Drop unnecessary columns
df_churn = df_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# One-hot encode categorical variables and convert to integer type
df_churn = pd.get_dummies(df_churn, drop_first=True)
df_churn = df_churn.astype(int)

# Visualize the target variable distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Exited', data=df_churn)
plt.title('Exited Distribution')
plt.show()

# Split the data into features and target variable
X_features = df_churn.drop('Exited', axis=1)
y_target = df_churn['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.1, random_state=42)
print('Training Shape:', X_train.shape)
print('Testing Shape:', X_test.shape)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
model_dict = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate each model
model_performance = []

for model_name, model in model_dict.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    model_performance.append({
        'Model': model_name,
        'Accuracy': accuracy
    })
    print(f"{model_name} Accuracy: {accuracy}")

# Create a performance summary DataFrame
performance_df = pd.DataFrame(model_performance)

# Display the performance summary
print(performance_df)
