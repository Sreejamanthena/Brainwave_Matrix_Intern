def predict_news(news_text, model, vectorizer):
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)
    return "REAL" if prediction[0] == 1 else "FAKE"
