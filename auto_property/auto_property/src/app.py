"""
Streamlit App for Auto Property

This module implements a Streamlit web interface for the property extraction system.
It allows users to upload documents, choose between RAG and SLM extraction methods,
and view the extracted property-value pairs.

Usage:
    streamlit run app.py
"""

import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional

# Import extraction methods
try:
    # Try relative imports first (when running from src directory)
    from data_preprocessing import clean_text, chunk_text
    from retriever_rag import extract_properties as rag_extract
    from slm_extractor import extract_properties as slm_extract
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from auto_property.src.data_preprocessing import clean_text, chunk_text
    from auto_property.src.retriever_rag import extract_properties as rag_extract
    from auto_property.src.slm_extractor import extract_properties as slm_extract


def setup_page():
    """Set up the Streamlit page with title and description."""
    st.set_page_config(
        page_title="Auto Property Extractor",
        page_icon="📄",
        layout="wide"
    )

    st.title("Auto Property Extractor")
    st.markdown("""
    Upload a document and extract property-value pairs using either:
    - **RAG**: Retrieval-Augmented Generation with LangChain
    - **SLM**: Fine-tuned Small Language Model
    """)


def upload_document() -> Optional[str]:
    """
    Allow the user to upload a document.

    Returns:
        Document text if uploaded, None otherwise
    """
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])

    if uploaded_file is not None:
        # Read the file
        if uploaded_file.name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
            return text
        elif uploaded_file.name.endswith('.json'):
            try:
                data = json.loads(uploaded_file.read().decode('utf-8'))
                # Extract text from JSON (simplified version)
                if isinstance(data, str):
                    return data
                elif isinstance(data, dict):
                    # Try to extract text from common fields
                    text_parts = []
                    for key in ['text', 'description', 'content']:
                        if key in data and isinstance(data[key], str):
                            text_parts.append(data[key])

                    # If we found text in common fields, return it
                    if text_parts:
                        return ' '.join(text_parts)

                    # Otherwise, convert the entire JSON to string
                    return json.dumps(data)
                else:
                    return json.dumps(data)
            except json.JSONDecodeError:
                st.error("Invalid JSON file")
                return None

    return None


def select_extraction_method() -> str:
    """
    Allow the user to select an extraction method.

    Returns:
        Selected method name ("rag" or "slm")
    """
    method = st.radio(
        "Select extraction method",
        ["RAG (Retrieval-Augmented Generation)", "SLM (Small Language Model)"]
    )

    # Return the method name in lowercase
    if "RAG" in method:
        return "rag"
    else:
        return "slm"


def extract_properties(text: str, method: str) -> Dict[str, str]:
    """
    Extract property-value pairs using the selected method.

    Args:
        text: Document text
        method: Extraction method ("rag" or "slm")

    Returns:
        Dictionary of property-value pairs
    """
    # Clean and preprocess the text
    cleaned_text = clean_text(text)
    chunks = chunk_text(cleaned_text)

    # Display method being used
    st.info(f"Using {method.upper()} method for extraction")

    # Extract properties using the selected method
    properties = {}

    if method == "rag":
        st.warning("RAG implementation is a placeholder. Using simplified extraction.")
        # Call the RAG extraction method
        rag_properties = rag_extract(cleaned_text)
        properties.update(rag_properties)

        # Also use pattern matching for better results in the demo
        pattern_properties = extract_properties_with_patterns(cleaned_text)
        properties.update(pattern_properties)
    else:  # method == "slm"
        st.warning("SLM implementation is a placeholder. Using simplified extraction.")
        # Call the SLM extraction method
        slm_properties = slm_extract(cleaned_text)
        properties.update(slm_properties)

        # Also use pattern matching for better results in the demo
        pattern_properties = extract_properties_with_patterns(cleaned_text)
        properties.update(pattern_properties)

    return properties


def extract_properties_with_patterns(text: str) -> Dict[str, str]:
    """
    Extract property-value pairs using simple pattern matching.
    This is a placeholder for demonstration purposes.

    Args:
        text: Cleaned text

    Returns:
        Dictionary of property-value pairs
    """
    import re

    properties = {}

    # Look for common patterns like "property: value" or "property - value"
    patterns = [
        r'(\w+[\s\w]*?):\s*([\w\s.,;!?()-]+)',  # property: value
        r'(\w+[\s\w]*?)\s*-\s*([\w\s.,;!?()-]+)',  # property - value
        r'(\w+[\s\w]*?)\s*=\s*([\w\s.,;!?()-]+)',  # property = value
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for prop, val in matches:
            prop = prop.strip().lower()
            val = val.strip()
            if prop and val:
                properties[prop] = val

    # Look for specific properties by keywords
    property_keywords = {
        "price": [r'price[:\s]+\$?([\d.,]+)', r'\$([\d.,]+)'],
        "weight": [r'weight[:\s]+([\d.,]+\s*(?:kg|g|lbs|pounds|oz|ounces))'],
        "dimensions": [r'dimensions[:\s]+([\d.,]+\s*[xX]\s*[\d.,]+\s*[xX]\s*[\d.,]+\s*(?:cm|mm|in|inches|ft|feet)?)'],
        "color": [r'colou?r[:\s]+([\w\s]+)'],
        "material": [r'material[:\s]+([\w\s]+)'],
    }

    for prop, patterns in property_keywords.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                properties[prop] = match.group(1).strip()
                break

    return properties


def display_results(properties: Dict[str, str]):
    """
    Display the extracted property-value pairs.

    Args:
        properties: Dictionary of property-value pairs
    """
    if not properties:
        st.warning("No properties were extracted.")
        return

    st.subheader("Extracted Properties")

    # Convert dictionary to DataFrame for better display
    df = pd.DataFrame(list(properties.items()), columns=["Property", "Value"])

    # Display as a table
    st.table(df)

    # Add download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="extracted_properties.csv",
        mime="text/csv",
    )


def main():
    """Main function to run the Streamlit app."""
    setup_page()

    # Upload document
    document_text = upload_document()

    if document_text:
        # Select extraction method
        method = select_extraction_method()

        # Extract properties
        if st.button("Extract Properties"):
            with st.spinner("Extracting properties..."):
                properties = extract_properties(document_text, method)

            # Display results
            display_results(properties)


if __name__ == "__main__":
    main()
