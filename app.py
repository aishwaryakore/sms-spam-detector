import nltk
import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def transform_text(text):
    # step 1
    text = text.lower()

    # step 2
    text = nltk.word_tokenize(text)

    # step 3
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # step 4
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # step 5
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. preprocess
    transformed_text = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")