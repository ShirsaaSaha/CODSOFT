import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
df_train = pd.read_csv("train_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
df_test = pd.read_csv("test_data.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
df_test_solution = pd.read_csv("test_data_solution.txt", sep=':::', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])

# Display basic information about the datasets
print(df_train.head())
print(f'Train data shape: {df_train.shape}')
print(df_test.head())
print(f'Test data shape: {df_test.shape}')
print(df_test_solution.head())
print(f'Test solution data shape: {df_test_solution.shape}')

# Visualize the number of movies per genre
plt.figure(figsize=(20, 8))
sns.countplot(y=df_train['GENRE'], order=df_train['GENRE'].value_counts().index)
plt.title('Number of Movies per Genre')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()

# Visualize the description length by genre
df_train['DESCRIPTION_length'] = df_train['DESCRIPTION'].apply(len)
plt.figure(figsize=(15, 10))
sns.barplot(x='GENRE', y='DESCRIPTION_length', data=df_train)
plt.title('Description Length by Genre')
plt.xticks(rotation=45)
plt.xlabel('Genre')
plt.ylabel('Description Length')
plt.show()

# Visualize the top 10 most frequent genres
top_genres = df_train['GENRE'].value_counts().head(10)
plt.figure(figsize=(20, 10))
top_genres.plot(kind='barh', color='cyan')
plt.title('Top 10 Most Frequent Genres')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.gca().invert_yaxis()  # Invert y-axis to have the genre with the most movies at the top
plt.show()

# Handle any potential missing values
df_train['DESCRIPTION'].fillna("", inplace=True)
df_test['DESCRIPTION'].fillna("", inplace=True)

# Feature extraction using TF-IDF
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=100000)
X_train_tfidf = vectorizer_tfidf.fit_transform(df_train['DESCRIPTION'])
X_test_tfidf = vectorizer_tfidf.transform(df_test['DESCRIPTION'])

# Encode the genre labels
encoder_label = LabelEncoder()
y_train_encoded = encoder_label.fit_transform(df_train['GENRE'])
y_test_encoded = encoder_label.transform(df_test_solution['GENRE'])

# Split the training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_tfidf, y_train_encoded, test_size=0.2, random_state=42)

# Train and evaluate the LinearSVC model
model_svc = LinearSVC()
model_svc.fit(X_train_split, y_train_split)

y_val_pred_svc = model_svc.predict(X_val_split)
print("Validation Accuracy (SVC):", accuracy_score(y_val_split, y_val_pred_svc))
print("Validation Classification Report (SVC):\n", classification_report(y_val_split, y_val_pred_svc))

y_test_pred_svc = model_svc.predict(X_test_tfidf)
print("Test Accuracy (SVC):", accuracy_score(y_test_encoded, y_test_pred_svc))
print("Test Classification Report (SVC):\n", classification_report(y_test_encoded, y_test_pred_svc))

# Train and evaluate the MultinomialNB model
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train_encoded)

y_test_pred_nb = model_nb.predict(X_test_tfidf)
print("Test Accuracy (NB):", accuracy_score(y_test_encoded, y_test_pred_nb))
print("Test Classification Report (NB):\n", classification_report(y_test_encoded, y_test_pred_nb))

# Train and evaluate the LogisticRegression model
model_lr = LogisticRegression(max_iter=500)
model_lr.fit(X_train_tfidf, y_train_encoded)

y_test_pred_lr = model_lr.predict(X_test_tfidf)
print("Test Accuracy (LR):", accuracy_score(y_test_encoded, y_test_pred_lr))
print("Test Classification Report (LR):\n", classification_report(y_test_encoded, y_test_pred_lr))

# Function to predict the genre of a given movie description
def predict_genre(description, model=model_svc):
    transformed_desc = vectorizer_tfidf.transform([description])
    predicted_label = model.predict(transformed_desc)
    return encoder_label.inverse_transform(predicted_label)[0]

# Example predictions
example_description_1 = "A movie where police chases the criminal and shoot him"
print(predict_genre(example_description_1))

example_description_2 = "A movie where a person chases a girl to get married with her but she refuses him."
print(predict_genre(example_description_2))
