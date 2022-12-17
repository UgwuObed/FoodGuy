import spacy
import re
import requests
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
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

def stem_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Stem the tokens
    stemmed_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the stemmed tokens back into a single string
    stemmed_text = " ".join(stemmed_tokens)

    return stemmed_text

# Stem the cleaned data
stemmed_data = [stem_text(text) for text in cleaned_data]

# Create the vocabulary list
vocab_list = []
for text in stemmed_data:
    voc

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Generate the TF-IDF vectors
vectors = vectorizer.fit_transform(cleaned_data)

# Create a vocabulary index
vocab_index = {word: index for index, word in enumerate(vocab_list)}

def one_hot_encode(text, vocab_index, max_length):
    # Tokenize the text
    tokens = nlp(text)
    
    # Initialize the one-hot encoded vector
    one_hot = np.zeros(max_length, dtype=int)
    
    # Set the indices corresponding to the words in the text to 1
    for token in tokens:
        if token.text in vocab_index:
            one_hot[vocab_index[token.text]] = 1
    
    return one_hot

# Set the maximum length of the one-hot encoded vectors
max_length = len(vocab_index)

# One-hot encode the stemmed data
one_hot_data = [one_hot_encode(text, vocab_index, max_length) for text in stemmed_data]

X = cleaned_data  # input data
y = labels  # labels or target values

# Split the dataset into a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the test set
accuracy = model.score(X_test, y_test)


