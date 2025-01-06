import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from langdetect import detect
import PyPDF2
import os

# Set up the Streamlit app
st.title("AI-Powered Text Analysis")

# Function to process text with sentiment analysis
def analyze_text(text):
    # Language detection
    lang = detect(text)
    st.write(f"Detected language: {lang}")
    
    # Sentiment analysis using TextBlob
    blob = TextBlob(text)
    st.write("Text Sentiment:")
    st.write(f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}")
    
    # Summarization using Hugging Face pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    st.write("Summary:")
    st.write(summary[0]["summary_text"])

# File upload section
uploaded_file = st.file_uploader("Upload a PDF or Text File", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        # Extract text from PDF
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    else:
        # Read text file
        text = uploaded_file.read().decode("utf-8")
    
    # Display the extracted text
    st.write("Uploaded Text:")
    st.write(text)
    
    # Analyze the text
    analyze_text(text)

# Example input
st.write("Or enter text manually below:")
user_text = st.text_area("Enter text:")
if st.button("Analyze Text"):
    analyze_text(user_text)
