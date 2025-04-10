import streamlit as st
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
from googletrans import Translator
import pandas as pd
import torch  # Ensure torch is imported

# Set the page title
st.set_page_config(page_title="ILR Language Assessment Tool", page_icon="🌍")

# Add a developer credit centered
st.markdown(
    """
    <h2 style='text-align: center;'>ILR Language Assessment Tool</h2>
    <h5 style='text-align: center;'>Developed by Dr. Kamal Tabine</h5>
    """,
    unsafe_allow_html=True,
)

# Load the tokenizer and model for ILR level prediction from Hugging Face Hub or local directory
model_name = "xlm-roberta-base"  # Use the Hugging Face model for base
fine_tuned_model_path = "huggingface/your_username/your_model_name"  # Use Hugging Face repo URL if hosted

# Load tokenizer and model
try:
    tokenizer = XLMRobertaTokenizer.from_pretrained(fine_tuned_model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(fine_tuned_model_path)
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")

# Function to predict the ILR level
def predict_ilr_level(text):
    try:
        encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.item()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Function to translate text if it's not in English
def translate_text(text, target_language="en"):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

# Function to extract main idea (simplified for demo)
def extract_main_idea(text):
    return "This is a summary of the main idea of the text."

# Function to extract supporting details (simplified for demo)
def extract_supporting_details(text):
    return ["Supporting detail 1", "Supporting detail 2", "Supporting detail 3"]

# Function to extract relevant vocabulary with translations (simplified for demo)
def extract_vocabulary(text):
    vocab = {
        "term1": "translation1",
        "term2": "translation2",
        "term3": "translation3",
    }
    return vocab

# Set up UI components with text area centered
st.markdown("<h4 style='text-align: center;'>Enter the text for analysis:</h4>", unsafe_allow_html=True)
input_text = st.text_area("", "", height=300, max_chars=1000)

# Center the 'Analyze' button
st.markdown(
    """
    <div style="text-align: center;">
    <button>Analyze</button>
    </div>
    """, unsafe_allow_html=True
)

if st.button("Analyze"):
    if input_text:
        # Translate text if it's not in English
        translated_text = translate_text(input_text)
        st.write(f"**Translated Text:**\n{translated_text}")

        # Predict ILR Level
        ilr_level = predict_ilr_level(translated_text)
        if ilr_level is not None:
            st.write(f"**Predicted ILR Level:** {ilr_level}")

        # Extract Main Idea
        main_idea = extract_main_idea(translated_text)
        st.write(f"**Main Idea:**\n{main_idea}")

        # Extract Supporting Details
        supporting_details = extract_supporting_details(translated_text)
        st.write("**Supporting Details:**")
        for detail in supporting_details:
            st.write(f"- {detail}")

        # Vocabulary Table
        vocab = extract_vocabulary(translated_text)
        vocab_df = pd.DataFrame(list(vocab.items()), columns=["Vocabulary Term", "English Translation"])
        st.write("**Vocabulary Table:**")
        st.dataframe(vocab_df)

        # ILR Level Justification
        st.write(f"**ILR Level Justification:**")
        st.write(f"The predicted ILR Level for this text is {ilr_level}. This classification is based on factors such as complexity, grammar, and vocabulary usage in the text.")
    else:
        st.write("Please enter some text to analyze.")
