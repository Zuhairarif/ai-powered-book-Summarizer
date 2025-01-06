import nltk
import os
import fitz  # PyMuPDF for PDF handling
from transformers import pipeline
import streamlit as st
from langdetect import detect
from textblob import TextBlob
from googletrans import Translator
import PyMuPDF


# Ensure required NLTK data is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Initialize Summarization Pipelines
hugging_face_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = Translator()

# Function to ensure temp directory exists
def ensure_temp_directory():
    """Creates a temporary directory if it doesn't already exist."""
    if not os.path.exists("temp"):
        os.makedirs("temp")

# Functions for Text Extraction
def extract_text_from_txt(file_path):
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading .txt file: {str(e)}")
        return ""

def extract_text_from_pdf(file_path):
    """Extracts text from a .pdf file."""
    try:
        with fitz.open(file_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        return text.replace("\n", " ").strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

# Language Detection
def detect_language(text):
    """Detects the language of the input text."""
    try:
        return detect(text)
    except Exception as e:
        st.warning(f"Could not detect language: {str(e)}")
        return "unknown"

# Translation
def translate_text(text, target_language='en'):
    """Translates text to the specified target language."""
    try:
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text

# Sentiment Analysis
def sentiment_analysis(text):
    """Performs sentiment analysis on the text."""
    blob = TextBlob(text)
    return blob.sentiment

# Custom Tokenizer Function (avoiding nltk's sent_tokenize)
import re

def custom_sent_tokenize(text):
    """Custom sentence tokenizer using regular expressions."""
    sentence_endings = re.compile(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    sentences = sentence_endings.split(text)
    return sentences

# Summarization
def chunk_text(text, chunk_size=1000):
    """Divides text into chunks for summarization."""
    sentences = custom_sent_tokenize(text)  # Use custom tokenizer
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk.strip())

    return chunks

def summarize_text(text, model="hugging_face", length="medium"):
    """Summarizes the input text using the selected model."""
    max_length = 130 if length == 'brief' else 300 if length == 'detailed' else 200
    min_length = 30
    chunks = chunk_text(text, chunk_size=800)  # Adjust chunk size for more granular summaries
    summaries = []

    if model == "hugging_face":
        for chunk in chunks:
            try:
                summary = hugging_face_summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                summaries.append(f"[Error summarizing chunk]: {str(e)}")

    return " ".join(summaries)

# Streamlit App Configuration
st.set_page_config(page_title="AI-Powered Book Summarizer", layout="wide")

# Header with Hamburger Menu
st.markdown(
    """
    <style>
        .css-18e3th9 {
            padding-top: 1.5rem;
        }
        .css-1d391kg {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .css-hxt7ib {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #hamburger {
            font-size: 1.8em;
            cursor: pointer;
            margin-right: 10px;
        }
    </style>
    <div class="css-hxt7ib">
        <div id="hamburger">☰</div>
        <h1 style="text-align:center;">📚 AI-Powered Book Summarizer</h1>
        <div>Developed by <b>Zuhair Arif</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Inputs
st.sidebar.markdown("### Upload your file (.txt or .pdf):")
uploaded_file = st.sidebar.file_uploader("Select a file", type=["txt", "pdf"])
st.sidebar.markdown("### Choose summary length:")
summary_length = st.sidebar.radio("Summary Length:", ["brief", "medium", "detailed"], index=1)
st.sidebar.markdown("### Select summarization model:")
selected_model = st.sidebar.selectbox("Summarization Model:", ["hugging_face"])

# Process Uploaded File
if uploaded_file is not None:
    # Save uploaded file
    ensure_temp_directory()
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract Text
    if uploaded_file.name.endswith('.txt'):
        text = extract_text_from_txt(file_path)
    elif uploaded_file.name.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        st.error("Unsupported file type.")
        text = None

    if text:
        # Language Detection
        language = detect_language(text)
        st.sidebar.markdown(f"**📖 Detected Language:** `{language.upper()}`")

        # Translate Text (if necessary)
        if language != "en":
            st.sidebar.markdown("🔄 Translating text to English...")
            text = translate_text(text)

        # Display Text Preview
        st.subheader("Extracted Text Preview:")
        preview_length = min(1000, len(text))  # Limit preview to 1000 characters
        st.text_area("Preview:", text[:preview_length], height=200)
        if st.checkbox("View Full Text"):
            st.text_area("Full Text:", text, height=400)

        # Sentiment Analysis
        sentiment = sentiment_analysis(text)
        st.sidebar.markdown("### Sentiment Analysis:")
        st.sidebar.markdown(f"- **Polarity:** {sentiment.polarity:.2f} (negative to positive scale)")
        st.sidebar.markdown(f"- **Subjectivity:** {sentiment.subjectivity:.2f} (fact to opinion scale)")

        # Generate Summary
        if st.button("Generate Summary"):
            with st.spinner("Summarizing... Please wait."):
                summary = summarize_text(text, model=selected_model, length=summary_length)
            st.subheader("Summarized Content:")
            st.write(summary)
    else:
        st.error("No text could be extracted from the uploaded file.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Zuhair Arif**")
  
