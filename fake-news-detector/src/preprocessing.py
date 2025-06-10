import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string
stop_words = stopwords.words("english")

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)
def load_and_preprocess_data():
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")
    fake['label'] = 0
    true['label'] = 1

    df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

    print("📊 Label counts:\n", df['label'].value_counts())

    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['label']

    return X, y

def vectorize_data(X):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)
    return X_vec, vectorizer
