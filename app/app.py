import sys
import os
import re
import streamlit as st
from src.inference import summarize_text

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Streamlit app layout configuration
st.set_page_config(page_title="SumItUp: Text Summarization Tool", layout="wide")

# Function to validate dialogue format
def is_valid_dialogue(text):
    """
    Validates if the input text is in a dialogue format.
    A valid dialogue contains colon-separated speakers (e.g., "Alice: Hello").
    """
    pattern = r"^\s*[a-zA-Z]+:\s+.*"  # Matches "Speaker: Message"
    lines = text.strip().split("\n")
    for line in lines:
        if not re.match(pattern, line):
            return False
    return True

# Add custom CSS with updated styles
st.markdown(
    """
    <style>
    /* General body styling */
    body {
        margin: 0;
        padding: 0;
        background: #0f2027; /* Dark blue */
        color: white;
        font-family: 'Arial', sans-serif;
    }

    /* Animated neural network background */
    .neural-background {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background: radial-gradient(circle, #003973, #0f2027);
        animation: pulse 10s infinite alternate;
    }

    @keyframes pulse {
        from {
            background: radial-gradient(circle, #003973, #0f2027);
        }
        to {
            background: radial-gradient(circle, #004e89, #00263b);
        }
    }

    /* Header styling */
    .header {
        text-align: center;
        padding: 20px;
        font-size: 32px;
        font-weight: bold;
        color: #0077be;
        text-shadow: 0 0 10px #00c8ff;
    }

    /* Input box styling */
    textarea {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        border-radius: 10px;
        border: 2px solid #0077be;
        background: #001f33;
        color: white;
        padding: 10px;
    }

    /* Sidebar styling */
    .sidebar .block-container {
        background: #003973;
        color: white;
        border: 2px solid #0077be;
        border-radius: 10px;
        padding: 10px;
    }

    /* Slider handle (toggle bar) styling */
    .stSlider > div > div > div > div {
        background: #0077be !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #0077be, #004e89);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 2px solid #0077be;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #004e89, #0077be);
    }

    /* Summary box styling */
    .summary-box {
        background: linear-gradient(135deg, #0077be, #004e89);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 16px;
        font-family: 'Arial', sans-serif;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border: 2px solid #0077be;
    }
    </style>
    <div class="neural-background"></div>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header">
        SumItUp: Text Summarization Tool
    </div>
    """,
    unsafe_allow_html=True,
)

# Input section
dialogue_input = st.text_area(
    "Enter Dialogue", 
    placeholder="Type your dialogue here... (e.g., Alice: Hello, how are you?)",
    height=200,
)

# Sidebar customization options
st.sidebar.header("Customize Output")
max_length = st.sidebar.slider("Maximum Summary Length", 50, 300, 64)
min_length = st.sidebar.slider("Minimum Summary Length", 10, 100, 10)
do_sample = st.sidebar.checkbox("Enable Sampling for Randomness", value=False)

# Summarize button
if st.button("🔍 Summarize"):
    if dialogue_input.strip():
        if not is_valid_dialogue(dialogue_input):
            st.error("❗ Please enter a valid dialogue. Use the format: Speaker: Message")
        else:
            summary = summarize_text(
                dialogue_input, max_length=max_length, min_length=min_length, do_sample=do_sample
            )

            # Display summary
            st.markdown("### 📜 **Generated Summary**")
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
    else:
        st.error("❗ Please enter some dialogue to summarize.")
