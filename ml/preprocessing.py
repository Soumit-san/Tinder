import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis (replace with empty string)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

def preprocess_pipeline(text):
    text = clean_text(text)
    
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
        
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 0:
            filtered_tokens.append(lemmatizer.lemmatize(token))
    
    # Filter short reviews (< 5 tokens)
    if len(filtered_tokens) < 5:
        return ""
        
    return ' '.join(filtered_tokens)

def main():
    print("Loading raw_reviews.csv...")
    try:
        df = pd.read_csv('data/raw_reviews.csv')
    except FileNotFoundError:
        print("Error: data/raw_reviews.csv not found. Run scraper.py first.")
        return
        
    print("Preprocessing texts (this may take a minute)...")
    df['clean_text'] = df['review_text'].apply(preprocess_pipeline)
    
    # Clean rows that were filtered out
    initial_len = len(df)
    df = df[df['clean_text'] != ""]
    print(f"Filtered out {initial_len - len(df)} reviews with < 5 tokens.")
    
    df.to_csv('data/clean_reviews.csv', index=False)
    print(f"Saved {len(df)} cleaned reviews to data/clean_reviews.csv")

if __name__ == "__main__":
    main()
