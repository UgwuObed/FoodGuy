import spacy
import re
import requests
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

# Print the cleaned data
print(cleaned_data)
