import os
import numpy as np
import pandas as pd
import json


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


import re


#%%

# ******************
#     Read data
# ******************
def read_json_folder(folder_path):
    """
    Read JSON files from a folder and return a list of dictionaries.
    Args:
        folder_path (str): Path to the folder containing JSON files.
    Returns:
        list: A list of dictionaries containing data from each JSON file.
    """
    json_data_list = []

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return json_data_list

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                # Load JSON data from the file
                try:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {file_path}")
                    continue

    df = pd.DataFrame.from_dict(json_data_list)

    return df, json_data_list

df, json_data_list = read_json_folder('Group Project/data/jsons')





#%%
# ******************
#   Preprocessing
# ******************

nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r"[^\w\s]", ' ', text)
    # Remove stopwords
    word_tokens = casual_tokenize(text)
    filtered_text = [w for w in word_tokens if w not in stop_words]
    text = ' '.join(filtered_text)
    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in casual_tokenize(text)]
    return ' '.join(lemmatized)


df['content_clean'] = df['content'].apply(preprocess)
#%%

def split(df):
    X = df.drop(columns=['topic', 'source', 'url',
                         'title', 'date', 'authors', 'content_original',
                         'source_url', 'bias_text', 'ID', 'bias'])
    y = df['bias']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)
    return X_train, X_test, y_train, y_test


df['content'] = [entry.lower() for entry in df['content']]
df['content'] = df['content'].apply(lambda x: re.sub(r"[^\w\s]", ' ', x).lower())


X_train, X_test, y_train, y_test = split(df)
#%%
# ******************
#   Model building
# ******************
def model_traning_logistic(X_train, y_train):
    vectorizer = TfidfVectorizer(tokenizer=casual_tokenize)
    X_train = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, vectorizer

def model_traning_naive(X_train, y_train):
    vectorizer_naive = TfidfVectorizer(tokenizer=casual_tokenize)
    X_train = vectorizer_naive.fit_transform(X_train)
    model_naive = MultinomialNB()
    model_naive.fit(X_train, y_train)
    return model_naive, vectorizer_naive


model, vectorizer = model_traning_logistic(X_train['content_clean'], y_train)
model_naive, vectorizer_naive = model_traning_naive(X_train['content_clean'], y_train)

def model_testing(model, vectorizer, X_test):
    X_test = vectorizer.transform(X_test)
    predictions = model.predict(X_test)
    return predictions


predictions = model_testing(model, vectorizer, X_test['content_clean'])
predictions_naive = model_testing(model_naive, vectorizer_naive, X_test['content_clean'])

#%%
# ******************
#   Metrics results
# ******************
def metrics(predictions, y_test):
    kappa = cohen_kappa_score(y_test, predictions)
    return kappa


print('*'*25)
print("Logistic Regression")
kappa = metrics(predictions, y_test)
print("Kappa score: ", kappa)

# Accuracy of the results
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# F1-score with 'weighted' averaging
f1_weighted = f1_score(y_test, predictions, average='weighted')
print("F1-score (weighted):", f1_weighted)

# F1-score with 'micro' averaging
f1_micro = f1_score(y_test, predictions, average='micro')
print("F1-score (micro):", f1_micro)

# F1-score with 'macro' averaging
f1_macro = f1_score(y_test, predictions, average='macro')
print("F1-score (macro):", f1_macro)

print()
print('*'*25)
print("Naive Regression")
kappa_naive = metrics(predictions_naive, y_test)
print("Kappa score: ", kappa_naive)

# Accuracy of the results
accuracy_naive = accuracy_score(y_test, predictions_naive)
print("Accuracy:", accuracy_naive)

# F1-score with 'weighted' averaging
f1_weighted_naive = f1_score(y_test, predictions_naive, average='weighted')
print("F1-score (weighted):", f1_weighted_naive)

# F1-score with 'micro' averaging
f1_micro_naive = f1_score(y_test, predictions_naive, average='micro')
print("F1-score (micro):", f1_micro_naive)

# F1-score with 'macro' averaging
f1_macro_naive = f1_score(y_test, predictions_naive, average='macro')
print("F1-score (macro):", f1_macro_naive)

#%%
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate confusion matrix for Logistic Regression
conf_matrix_logistic = confusion_matrix(y_test, predictions)

# Calculate confusion matrix for Naive Bayes
conf_matrix_naive = confusion_matrix(y_test, predictions_naive)

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_logistic, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'], cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_logistic.png', dpi=300)  # Save the plot as a PNG file with 300 DPI
plt.close()  # Close the plot to free up memory

# Plot confusion matrix for Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_naive, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'], cbar=False)
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_naive.png', dpi=300)  # Save the plot as a PNG file with 300 DPI
plt.close()  # Close the plot to free up memory

#%%
import joblib
import pickle

# Save logistic regression model and vectorizer
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Save naive Bayes model and vectorizer
with open('naive_bayes_model.pkl', 'wb') as file:
    pickle.dump(model_naive, file)

with open('tfidf_vectorizer_naive.pkl', 'wb') as file:
    pickle.dump(vectorizer_naive, file)

#%%
import pickle

# Load logistic regression model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as file:
    loaded_model_lr = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    loaded_vectorizer_lr = pickle.load(file)

# Load naive Bayes model and vectorizer
with open('naive_bayes_model.pkl', 'rb') as file:
    loaded_model_naive = pickle.load(file)

with open('tfidf_vectorizer_naive.pkl', 'rb') as file:
    loaded_vectorizer_naive = pickle.load(file)

