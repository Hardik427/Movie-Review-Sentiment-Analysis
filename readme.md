
# 🎬 IMDb Movie Review Sentiment Analysis

This is a Streamlit web app that uses a pre-trained RNN model to predict whether a given IMDb movie review is **positive** or **negative**.

## 🚀 Features

- Input raw text reviews
- Predict sentiment using a trained RNN model (`simple_RNN_imdb.h5`)
- See prediction confidence scores
- Simple UI using Streamlit

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Hardik427/imdb-sentiment-analysis.git
cd Movie-Review-Sentiment-Analysis
```

### 2. Install dependencies

Make sure you have Python 3.10 installed, then run:

```bash
pip install streamlit tensorflow
```

### 3. Add the model file

Place the trained model file `simple_RNN_imdb.h5` in the root of the project directory.

> If you don’t have the model file, you can train one using the IMDb dataset from Keras.

### 4. Run the app

```bash
streamlit run app.py
```

Go to `http://localhost:8501` in your browser.

## 🧠 Model

The app uses a simple RNN trained on the IMDb movie review dataset. Words are tokenized using `keras.datasets.imdb.get_word_index()` and padded to a maximum sequence length of 500.

## 📂 File Structure

```
.
├── app.py                # Streamlit app
├── simple_RNN_imdb.h5    # Pretrained RNN model (not included)
└── README.md             # Project documentation
```

## 📄 License

MIT License. Feel free to use and modify.
