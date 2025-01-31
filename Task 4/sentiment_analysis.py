import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load precomputed TF-IDF Features and Labels
tfidf_file = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/tfidf_features.csv'
labels_file = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/processed_sms_data.csv'

X_tfidf = pd.read_csv(tfidf_file)  # Load precomputed TF-IDF features
y = pd.read_csv(labels_file)['label'].map({'ham': 0, 'spam': 1})  # Load labels and map 'ham' to 0, 'spam' to 1

# Check for missing values in y
print('Missing values in y: ', y.isnull().sum())  # Check for missing values in the label column

# Drop rows where y is NaN
valid_indices = y.dropna().index
X_tfidf = X_tfidf.loc[valid_indices]
y = y.loc[valid_indices]

# Ensure no missing values in y
print("Missing values in y after dropping: ", y.isnull().sum())  # Should be 0

# STEP 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# STEP 3: Train and evaluate the Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# STEP 4: Evaluate Logistic Regression model
log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)

print('--- Logistic Regression Results ---')
print(f'Accuracy: {log_accuracy:.4f}')
print(f'Precision: {log_precision:.4f}')
print(f'Recall: {log_recall:.4f}')
print(f'F1-score: {log_f1:.4f}')

# STEP 5: Train and evaluate the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# STEP 6: Evaluate Naive Bayes model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)

print('--- Naive Bayes Results ---')
print(f'Accuracy: {nb_accuracy:.4f}')
print(f'Precision: {nb_precision:.4f}')
print(f'Recall: {nb_recall:.4f}')
print(f'F1-score: {nb_f1:.4f}')

# STEP 7: Confusion Matrix for both models
# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('logistic_regression_confusion_matrix.png')  # Save the Logistic Regression confusion matrix
plt.close()

# Naive Bayes Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('naive_bayes_confusion_matrix.png')  # Save the Naive Bayes confusion matrix
plt.close()