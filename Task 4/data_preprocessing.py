import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Step 1: Load the dataset
data = pd.read_csv('C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/spam_sms.csv', names=["label", "message"])

# Step 2: Clean the text data
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters (punctuation)
    
    return text

# Apply cleaning function
data['cleaned_message'] = data['message'].apply(clean_text)

# Step 3: Tokenize the text
# We'll use word_tokenize from nltk for tokenization
nltk.download('punkt')
data['tokens'] = data['cleaned_message'].apply(word_tokenize)

# Step 4: Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

data['tokens'] = data['tokens'].apply(remove_stopwords)

# Step 5: Apply Stemming or Lemmatization
# Stemming
stemmer = PorterStemmer()
data['stemmed_tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

# Alternatively, you can use Lemmatization
lemmatizer = WordNetLemmatizer()
data['lemmatized_tokens'] = data['tokens'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Preview the preprocessed data
print(data[['message', 'cleaned_message', 'tokens', 'stemmed_tokens', 'lemmatized_tokens']].head())

# Step 6: Save the processed data to a CSV file
processed_file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 4/processed_sms_data.csv'
data.to_csv(processed_file_path, index=False)