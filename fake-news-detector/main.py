from src.preprocessing import load_and_preprocess_data, vectorize_data
from src.model import train_model
from src.predict import predict_news

def main():
    print("🚀 Loading and processing data...")
    X, y = load_and_preprocess_data()


    print("🔠 Vectorizing text...")
    X_vectorized, tfidf_vectorizer = vectorize_data(X)

    print("🧠 Training model...")
    model = train_model(X_vectorized, y)

    # Try a custom prediction
    sample_text = "The government has launched a new initiative for climate change."
    print("\n🗞️ Sample News: ", sample_text)
    result = predict_news(sample_text, model, tfidf_vectorizer)
    print("🔍 Prediction:", result)

if __name__ == "__main__":
    main()
