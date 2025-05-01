"""
Test script for property extraction using both RAG and SLM approaches.

This script loads a sample document, extracts properties using both
the RAG and SLM approaches, and compares the results.
"""

import os
from src.data_preprocessing import load_document, clean_text, chunk_text
from src.retriever_rag import extract_properties as rag_extract
from src.slm_extractor import extract_properties as slm_extract

def test_extraction():
    """Test property extraction using both RAG and SLM approaches."""
    # Test with a text file
    print("Testing with a text file...")
    text_path = "data/sample_input.txt"
    if os.path.exists(text_path):
        # Load and preprocess the document
        text = load_document(text_path)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        
        print(f"Loaded document: {text_path}")
        print(f"Cleaned text (first 100 chars): {cleaned_text[:100]}...")
        print(f"Created {len(chunks)} chunks")
        
        # Extract properties using RAG
        print("\nExtracting properties using RAG...")
        rag_properties = rag_extract(cleaned_text)
        print("RAG properties:")
        for prop, value in rag_properties.items():
            print(f"  {prop}: {value}")
        
        # Extract properties using SLM
        print("\nExtracting properties using SLM...")
        slm_properties = slm_extract(cleaned_text)
        print("SLM properties:")
        for prop, value in slm_properties.items():
            print(f"  {prop}: {value}")
    else:
        print(f"File not found: {text_path}")
    
    # Test with a JSON file
    print("\nTesting with a JSON file...")
    json_path = "data/sample_input.json"
    if os.path.exists(json_path):
        # Load and preprocess the document
        text = load_document(json_path)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        
        print(f"Loaded document: {json_path}")
        print(f"Cleaned text (first 100 chars): {cleaned_text[:100]}...")
        print(f"Created {len(chunks)} chunks")
        
        # Extract properties using RAG
        print("\nExtracting properties using RAG...")
        rag_properties = rag_extract(cleaned_text)
        print("RAG properties:")
        for prop, value in rag_properties.items():
            print(f"  {prop}: {value}")
        
        # Extract properties using SLM
        print("\nExtracting properties using SLM...")
        slm_properties = slm_extract(cleaned_text)
        print("SLM properties:")
        for prop, value in slm_properties.items():
            print(f"  {prop}: {value}")
    else:
        print(f"File not found: {json_path}")

if __name__ == "__main__":
    test_extraction()