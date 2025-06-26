import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load IMDb word index and reverse it
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
model = load_model('simple_RNN_imdb.h5')

# Decode review from integers to text (optional utility)
def decode_review(encoded_review):
    return " ".join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

# Preprocess raw text input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Predict sentiment
def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

# Streamlit UI
st.title("IMDb Movie Review Sentiment Analysis")

user_review = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    sentiment, confidence = predict_sentiment(user_review)
    st.write(f"**Sentiment:** {sentiment.capitalize()}")
    st.write(f"**Confidence:** {confidence:.2f}")
