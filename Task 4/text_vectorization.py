from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import ast

# Step 1: Load the preprocessed data
data = pd.read_csv('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/processed_sms_data.csv')

# Convert string representation of list to actual list
data['lemmatized_tokens'] = data['lemmatized_tokens'].apply(ast.literal_eval)

# Now join the tokens
data['lemmatized_message'] = data['lemmatized_tokens'].apply(lambda x: ' '.join(x))

# Step 3: Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can set the number of features (words) to keep
X_tfidf = tfidf_vectorizer.fit_transform(data['lemmatized_message'])

# Step 4: Fit the TF-IDF Vectorizer on the lemmatized messages and transform them into numerical features
X_tfidf = tfidf_vectorizer.fit_transform(data['lemmatized_message'])

# Step 5: View the shape of the resulting TF-IDF matrix
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Step 6: Optionally, view the feature names (words used)
print(f"Feature names: {tfidf_vectorizer.get_feature_names_out()[:10]}")  # Print first 10 features for example

# Step 7: Save the transformed data (if needed)
# You can save the TF-IDF transformed data as a CSV (converted to DataFrame first)
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_tfidf_df.to_csv('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/tfidf_features.csv', index=False)