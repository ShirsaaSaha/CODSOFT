import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset
df_spam = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Display basic information about the dataset
print(df_spam.head())
print(df_spam.info())

# Drop unnecessary columns and rename the remaining columns
df_spam = df_spam.drop(columns=df_spam.columns[2:5])
df_spam.columns = ['Category', 'Message']
print(df_spam.head())

# Check for missing values
print(df_spam.isnull().sum())

# Visualize the category distribution
category_counts = df_spam['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Count', data=category_counts)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Distribution')

for i, count in enumerate(category_counts['Count']):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()

# Encode the target variable
df_spam['is_spam'] = df_spam['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(df_spam.head())

# Split the data into training and testing sets
X_train_msgs, X_test_msgs, y_train_labels, y_test_labels = train_test_split(df_spam.Message, df_spam.is_spam, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
spam_classifier_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model
spam_classifier_pipeline.fit(X_train_msgs, y_train_labels)

# Evaluate the model
print(f"Model Accuracy: {spam_classifier_pipeline.score(X_test_msgs, y_test_labels)}")

# Pre-trained model
nb_model = spam_classifier_pipeline.named_steps['nb']
vectorizer = spam_classifier_pipeline.named_steps['vectorizer']

# New sentences to predict
new_messages = [
    "Your account have 100 debeted, is waiting to be collected. Simply text the password \n to 85069 to verify. Get Usher and Britney. FML"
]

new_messages_count = vectorizer.transform(new_messages)

# Predict whether each sentence is spam (1) or not (0)
predictions = nb_model.predict(new_messages_count)

for message, prediction in zip(new_messages, predictions):
    if prediction == 1:
        print(f"'{message}' is a spam message.")
    else:
        print(f"'{message}' is not a spam message.")
