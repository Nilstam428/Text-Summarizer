import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

# Download necessary NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

def summarize(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    # Calculate word frequencies
    freq = FreqDist(words)
    
    # Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]
    
    # Get the top N sentences with highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    # Join the summary sentences
    summary = ' '.join(summary_sentences)
    
    return summary

# Streamlit interface
st.title('Text Summarization Tool')

# Text input
text = st.text_area("Enter the text you want to summarize:", height=200)

# Number of sentences slider
num_sentences = st.slider("Select number of sentences for the summary:", min_value=1, max_value=10, value=3)

if st.button('Summarize'):
    if text:
        summary = summarize(text, num_sentences)
        st.subheader("Summary:")
        st.write(summary)
        st.write(f"Original text length: {len(text)} characters")
        st.write(f"Summary length: {len(summary)} characters")
    else:
        st.warning("Please enter some text to summarize.")

# Instructions
st.sidebar.header("How to use:")
st.sidebar.write("""
1. Enter or paste your text in the text area.
2. Adjust the number of sentences you want in your summary using the slider.
3. Click the 'Summarize' button.
4. The summary will appear below the button.
""")

# About
st.sidebar.header("About:")
st.sidebar.write("""
This tool uses a simple extractive summarization technique. It scores sentences based on the frequency of their words and selects the top-ranked sentences to form the summary.

Note: This is a basic implementation and may not always produce perfect summaries, especially for complex texts.
""")