import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK resources (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopword set once
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Load the trained models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display result
    if result == 1:
        st.header("🚫 Spam")
    else:
        st.header("✅ Not Spam")
