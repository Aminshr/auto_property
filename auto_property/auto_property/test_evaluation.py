"""
Test script for the evaluation module.

This script tests the evaluation module by:
1. Preprocessing sample input files
2. Running evaluation on both RAG and SLM extraction methods
3. Comparing the results
"""

import os
import json
from src.data_preprocessing import load_document, clean_text
from src.evaluation import (
    load_ground_truth,
    evaluate_extraction,
    compare_methods,
    log_results
)
from src.retriever_rag import extract_properties as rag_extract
from src.slm_extractor import extract_properties as slm_extract

def main():
    # Paths to sample files
    sample_files = [
        "data/sample_input.txt",
        "data/large_sample_input.txt",
        "data/sample_input.json"
    ]
    ground_truth_path = "data/sample_ground_truth.json"
    output_path = "data/evaluation_results.csv"
    
    # Load ground truth data
    print(f"Loading ground truth data from {ground_truth_path}...")
    ground_truth = load_ground_truth(ground_truth_path)
    print(f"Loaded ground truth for {len(ground_truth)} documents")
    
    # Prepare test data
    test_data = {}
    for file_path in sample_files:
        if os.path.exists(file_path):
            # Load and clean the document
            text = load_document(file_path)
            cleaned_text = clean_text(text)
            
            # Use filename without extension as document ID
            doc_id = os.path.basename(file_path).split('.')[0]
            test_data[doc_id] = cleaned_text
            
            print(f"Loaded and cleaned {file_path}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"Prepared {len(test_data)} test documents")
    
    # Set up method arguments
    rag_args = {"query": "Extract all property-value pairs from this document"}
    slm_args = {}  # No specific arguments for SLM
    
    # Compare both methods
    print("\nComparing RAG and SLM extraction methods...")
    results = compare_methods(test_data, ground_truth, rag_args, slm_args)
    
    # Print the results
    print("\nEvaluation Results:")
    print("------------------")
    print(results.to_string(index=False))
    
    # Log the results
    log_results(results, output_path, save_details=True)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()