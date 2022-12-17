import spacy
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from spacy.lang.en.stop_words import STOP_WORDS

# Load the language model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and tokenize the text
    tokens = nlp(text.lower())
    
    # Remove stopwords and lemmatize the tokens
    tokens = [token.lemma_ for token in tokens if not token.is_stop]
    
    # Join the tokens back into a single string
    text = " ".join(tokens)
    
    return text

def create_vocab_list(text):
    # Tokenize the text
    tokens = nlp(text)
    
    # Remove stopwords and non-alphabetic tokens
    filtered_tokens = [token.text for token in tokens if not token.is_stop and token.is_alpha]
    
    # Remove duplicates and sort the list alphabetically
    vocab_list = sorted(set(filtered_tokens))
    
    return vocab_list

def one_hot_encode(text, vocab_index, max_length):
    # Tokenize the text
    tokens = nlp(text)
    
    # Initialize the one-hot encoded vector
    one_hot = np.zeros(max_length, dtype=int)
    
    # Set the indices corresponding to the words in the text to 1
    for token in tokens:
        if token.text in vocab_index:
            one_hot[vocab_index[token.text]] = 1
        else:
            # Use the default value for words that are not in the vocabulary index
            one_hot[vocab_index[-1]] = 1
    
    return one_hot

# Send a GET request to the webpage
response = requests.get("https://www.businesslist.com.ng/category/restaurants/city:lagos")

# Get the HTML content of the webpage
html = response.text

# Parse the HTML content of the webpage
soup = BeautifulSoup(html, "html.parser")

# Extract the names of the restaurants from the HTML
restaurant_names = soup.find_all("h2")

# Clean the dataset
cleaned_data = [clean_text(name.text) for name in restaurant_names]

# Remove duplicates from the dataset
cleaned_data = list(set(cleaned_data))

# Tokenize the cleaned data
tokenized_data = [nlp(text) for text in cleaned_data]

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize the tokens
lemmatized_data = [[lemmatizer.lemmatize(token.text) for token in doc] for doc in tokenized_data]

# Create a vocabulary list
vocab_list = create_vocab_list(lemmatized_data)

# Create a vocabulary index
vocab_index = {word: index for index, word in enumerate(vocab_list)}

# Set a default value for words that are not in the vocabulary index
vocab_index.setdefault(-1)

# One-hot encode the data
one_hot_data = [one_hot_encode(text, vocab_index, len(vocab_list)) for text in lemmatized_data]

# Get the maximum length of the one-hot encoded data
max_length = max([len(doc) for doc in one_hot_data])

# Pad the one-hot encoded data
padded_data = [np.pad(doc, (0, max_length - len(doc)), "constant") for doc in one_hot_data]

# Scale the data
scaled_data = scale(padded_data)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2)

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the train data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Create a dataframe from the one-hot encoded data
df = pd.DataFrame(one_hot_data)

# Create a heatmap
sns.heatmap(df.corr())

# Show the plot
plt.show()

