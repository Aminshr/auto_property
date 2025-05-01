"""
Data Preprocessing Module for Auto Property

This module provides functions for cleaning and preprocessing text data,
chunking it into manageable pieces, and saving the processed data in
a format suitable for further processing.

Functions:
    clean_text: Removes HTML tags, special characters, and normalizes text
    chunk_text: Splits text into chunks of approximately 100 tokens
    save_chunks_to_jsonl: Saves chunks to a JSONL file
    save_chunks_to_txt: Saves chunks to a TXT file for debugging
    load_document: Loads text from a file (.txt or .json)
"""

import os
import re
import json
import jsonlines
import nltk
from bs4 import BeautifulSoup
from typing import List, Dict, Union, Optional
import argparse

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, special characters, and normalizing.

    Args:
        text: Raw text string to be cleaned

    Returns:
        Cleaned text string
    """
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html5lib')
    text = soup.get_text(separator=' ')

    # Remove special characters and keep only alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """
    Split text into chunks of approximately chunk_size tokens.

    Args:
        text: Text string to be chunked
        chunk_size: Target number of tokens per chunk (default: 100)

    Returns:
        List of text chunks
    """
    # Simple tokenization by splitting on whitespace
    tokens = text.split()

    # Create chunks
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def save_chunks_to_jsonl(chunks: List[str], output_path: str) -> None:
    """
    Save chunks to a JSONL file with IDs.

    Args:
        chunks: List of text chunks
        output_path: Path to save the JSONL file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode='w') as writer:
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i:03d}"
            writer.write({"id": chunk_id, "text": chunk})

    print(f"Saved {len(chunks)} chunks to {output_path}")


def save_chunks_to_txt(chunks: List[str], output_path: str) -> None:
    """
    Save chunks to a TXT file for debugging.

    Args:
        chunks: List of text chunks
        output_path: Path to save the TXT file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i:03d} ---\n")
            f.write(chunk)
            f.write("\n\n")

    print(f"Saved {len(chunks)} chunks to {output_path} for debugging")


def load_document(file_path: str) -> str:
    """
    Load text from a file (.txt or .json).

    Args:
        file_path: Path to the input file

    Returns:
        Text content from the file
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, str):
                return data
            elif isinstance(data, dict) and 'text' in data:
                return data['text']
            elif isinstance(data, dict) and 'description' in data:
                return data['description']
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                return ' '.join(data)
            else:
                # Try to extract all string values and join them
                text_parts = []

                def extract_strings(obj):
                    if isinstance(obj, str):
                        text_parts.append(obj)
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            extract_strings(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_strings(item)

                extract_strings(data)
                return ' '.join(text_parts)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please provide a .txt or .json file.")


def process_document(input_path: str, output_jsonl_path: str, output_txt_path: Optional[str] = None, 
                    chunk_size: int = 100) -> None:
    """
    Process a document: load, clean, chunk, and save.

    Args:
        input_path: Path to the input file (.txt or .json)
        output_jsonl_path: Path to save the processed chunks as JSONL
        output_txt_path: Optional path to save the processed chunks as TXT for debugging
        chunk_size: Target number of tokens per chunk
    """
    # Load document
    print(f"Loading document from {input_path}...")
    text = load_document(input_path)

    # Clean text
    print("Cleaning text...")
    cleaned_text = clean_text(text)

    # Chunk text
    print(f"Chunking text into ~{chunk_size} token pieces...")
    chunks = chunk_text(cleaned_text, chunk_size)

    # Save chunks to JSONL
    save_chunks_to_jsonl(chunks, output_jsonl_path)

    # Optionally save chunks to TXT for debugging
    if output_txt_path:
        save_chunks_to_txt(chunks, output_txt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess documents for property extraction")
    parser.add_argument("input_path", help="Path to the input file (.txt or .json)")
    parser.add_argument("--output_jsonl", default="data/clean_chunks.jsonl", 
                        help="Path to save the processed chunks as JSONL")
    parser.add_argument("--output_txt", default=None, 
                        help="Optional path to save the processed chunks as TXT for debugging")
    parser.add_argument("--chunk_size", type=int, default=100, 
                        help="Target number of tokens per chunk")

    args = parser.parse_args()

    process_document(
        args.input_path,
        args.output_jsonl,
        args.output_txt,
        args.chunk_size
    )
