"""
Evaluation Module for Auto Property

This module provides functionality for evaluating and comparing the performance
of different property extraction methods (RAG vs SLM) using metrics like
precision, recall, and F1 score.

Functions:
    load_ground_truth: Loads ground truth property-value pairs
    evaluate_extraction: Evaluates extraction results against ground truth
    compare_methods: Compares different extraction methods
    calculate_metrics: Calculates precision, recall, and F1 score
    log_results: Logs evaluation results
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support

# Import extraction methods
from retriever_rag import extract_properties as rag_extract
from slm_extractor import extract_properties as slm_extract


# Placeholder for actual implementation
def load_ground_truth(ground_truth_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load ground truth property-value pairs from a file.

    Args:
        ground_truth_path: Path to the ground truth file

    Returns:
        Dictionary mapping document IDs to dictionaries of property-value pairs
    """
    # Check if the file exists
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    # Load the ground truth data based on file extension
    file_ext = os.path.splitext(ground_truth_path)[1].lower()

    if file_ext == '.json':
        # Load from JSON file
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
    elif file_ext == '.jsonl':
        # Load from JSONL file
        ground_truth = {}
        import jsonlines
        with jsonlines.open(ground_truth_path, mode='r') as reader:
            for item in reader:
                if 'id' in item and 'properties' in item:
                    ground_truth[item['id']] = item['properties']
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please provide a .json or .jsonl file.")

    # Validate the ground truth data
    if not ground_truth:
        raise ValueError("Ground truth data is empty")

    # Ensure all values are strings
    for doc_id, properties in ground_truth.items():
        ground_truth[doc_id] = {k: str(v) for k, v in properties.items()}

    return ground_truth


def calculate_metrics(
    predicted: Dict[str, str],
    ground_truth: Dict[str, str]
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for property extraction.

    Args:
        predicted: Dictionary of predicted property-value pairs
        ground_truth: Dictionary of ground truth property-value pairs

    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Normalize keys for case-insensitive comparison
    norm_predicted = {k.lower().strip(): v for k, v in predicted.items()}
    norm_ground_truth = {k.lower().strip(): v for k, v in ground_truth.items()}

    # Calculate true positives and false positives
    for key, value in norm_predicted.items():
        if key in norm_ground_truth:
            # Key exists in ground truth
            gt_value = norm_ground_truth[key]
            # Compare values (case-insensitive)
            if value.lower().strip() == gt_value.lower().strip():
                true_positives += 1
            else:
                # Key exists but value is wrong
                false_positives += 1
        else:
            # Key doesn't exist in ground truth
            false_positives += 1

    # Calculate false negatives (keys in ground truth but not in predicted)
    for key in norm_ground_truth:
        if key not in norm_predicted:
            false_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def evaluate_extraction(
    extraction_method: callable,
    test_data: Dict[str, str],
    ground_truth: Dict[str, Dict[str, str]],
    method_args: Dict[str, Any] = {},
    measure_latency: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an extraction method on test data.

    Args:
        extraction_method: Function that extracts property-value pairs
        test_data: Dictionary mapping document IDs to text content
        ground_truth: Dictionary mapping document IDs to ground truth property-value pairs
        method_args: Additional arguments for the extraction method
        measure_latency: Whether to measure extraction latency

    Returns:
        Dictionary containing evaluation results
    """
    results = {
        "per_document": {},
        "overall": {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "latency": 0.0
        }
    }

    total_docs = len(test_data)
    if total_docs == 0:
        return results

    # Track overall metrics
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_latency = 0.0

    # Process each document
    for doc_id, text in test_data.items():
        # Skip if no ground truth for this document
        if doc_id not in ground_truth:
            print(f"Warning: No ground truth for document {doc_id}, skipping")
            continue

        doc_ground_truth = ground_truth[doc_id]

        # Extract properties with timing
        start_time = time.time()
        try:
            predicted_properties = extraction_method(text, **method_args)
        except Exception as e:
            print(f"Error extracting properties from document {doc_id}: {e}")
            predicted_properties = {}
        end_time = time.time()

        # Calculate latency
        latency = end_time - start_time if measure_latency else 0

        # Calculate metrics for this document
        metrics = calculate_metrics(predicted_properties, doc_ground_truth)

        # Store results for this document
        results["per_document"][doc_id] = {
            "metrics": metrics,
            "latency": latency,
            "predicted": predicted_properties
        }

        # Update totals
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1"]
        total_latency += latency

    # Calculate averages
    valid_docs = len(results["per_document"])
    if valid_docs > 0:
        results["overall"]["precision"] = total_precision / valid_docs
        results["overall"]["recall"] = total_recall / valid_docs
        results["overall"]["f1"] = total_f1 / valid_docs
        results["overall"]["latency"] = total_latency / valid_docs

    return results


def compare_methods(
    test_data: Dict[str, str],
    ground_truth: Dict[str, Dict[str, str]],
    rag_args: Dict[str, Any] = {},
    slm_args: Dict[str, Any] = {}
) -> pd.DataFrame:
    """
    Compare RAG and SLM extraction methods.

    Args:
        test_data: Dictionary mapping document IDs to text content
        ground_truth: Dictionary mapping document IDs to ground truth property-value pairs
        rag_args: Additional arguments for the RAG extraction method
        slm_args: Additional arguments for the SLM extraction method

    Returns:
        DataFrame comparing the performance of both methods
    """
    print("Evaluating RAG extraction method...")
    rag_results = evaluate_extraction(rag_extract, test_data, ground_truth, rag_args)

    print("Evaluating SLM extraction method...")
    slm_results = evaluate_extraction(slm_extract, test_data, ground_truth, slm_args)

    # Create a comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Latency (s)'],
        'RAG': [
            rag_results['overall']['precision'],
            rag_results['overall']['recall'],
            rag_results['overall']['f1'],
            rag_results['overall']['latency']
        ],
        'SLM': [
            slm_results['overall']['precision'],
            slm_results['overall']['recall'],
            slm_results['overall']['f1'],
            slm_results['overall']['latency']
        ]
    })

    # Calculate the difference (SLM - RAG)
    comparison['Difference'] = comparison['SLM'] - comparison['RAG']

    # Add per-document comparison
    doc_comparisons = []
    for doc_id in test_data.keys():
        if doc_id in rag_results['per_document'] and doc_id in slm_results['per_document']:
            rag_f1 = rag_results['per_document'][doc_id]['metrics']['f1']
            slm_f1 = slm_results['per_document'][doc_id]['metrics']['f1']
            better_method = 'SLM' if slm_f1 > rag_f1 else 'RAG' if rag_f1 > slm_f1 else 'Tie'
            doc_comparisons.append({
                'Document': doc_id,
                'RAG_F1': rag_f1,
                'SLM_F1': slm_f1,
                'Difference': slm_f1 - rag_f1,
                'Better_Method': better_method
            })

    # Create a per-document comparison DataFrame
    if doc_comparisons:
        doc_comparison_df = pd.DataFrame(doc_comparisons)

        # Count how many times each method performed better
        method_counts = doc_comparison_df['Better_Method'].value_counts().to_dict()

        # Add summary row to the main comparison
        comparison.loc[len(comparison)] = [
            'Documents where better',
            method_counts.get('RAG', 0),
            method_counts.get('SLM', 0),
            method_counts.get('SLM', 0) - method_counts.get('RAG', 0)
        ]

    # Store the detailed results for potential further analysis
    comparison.attrs['rag_results'] = rag_results
    comparison.attrs['slm_results'] = slm_results

    return comparison


def log_results(results: pd.DataFrame, output_path: str, save_details: bool = False) -> None:
    """
    Log evaluation results to a file.

    Args:
        results: DataFrame containing evaluation results
        output_path: Path to save the results
        save_details: Whether to save detailed results
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Determine the file format based on the extension
    file_ext = os.path.splitext(output_path)[1].lower()

    # Save the results
    if file_ext == '.csv':
        results.to_csv(output_path, index=False)
        print(f"Results saved to CSV: {output_path}")
    elif file_ext == '.json':
        # Convert to JSON
        results_json = results.to_dict(orient='records')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to JSON: {output_path}")
    else:
        # Default to CSV
        csv_path = f"{os.path.splitext(output_path)[0]}.csv"
        results.to_csv(csv_path, index=False)
        print(f"Results saved to CSV: {csv_path}")

    # Save detailed results if requested
    if save_details and hasattr(results, 'attrs') and 'rag_results' in results.attrs and 'slm_results' in results.attrs:
        # Create a details directory
        details_dir = os.path.join(output_dir, 'details')
        os.makedirs(details_dir, exist_ok=True)

        # Save RAG results
        rag_path = os.path.join(details_dir, 'rag_results.json')
        with open(rag_path, 'w', encoding='utf-8') as f:
            json.dump(results.attrs['rag_results'], f, indent=2)
        print(f"RAG details saved to: {rag_path}")

        # Save SLM results
        slm_path = os.path.join(details_dir, 'slm_results.json')
        with open(slm_path, 'w', encoding='utf-8') as f:
            json.dump(results.attrs['slm_results'], f, indent=2)
        print(f"SLM details saved to: {slm_path}")

        # Save per-document comparison if available
        if 'doc_comparison_df' in locals():
            doc_comp_path = os.path.join(details_dir, 'document_comparison.csv')
            doc_comparison_df.to_csv(doc_comp_path, index=False)
            print(f"Document comparison saved to: {doc_comp_path}")

    print(f"Evaluation results successfully logged to {output_path}")


if __name__ == "__main__":
    import argparse
    import jsonlines
    from data_preprocessing import load_document, clean_text

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate and compare property extraction methods")

    # Input data arguments
    parser.add_argument("--test-data", "-t", required=True,
                        help="Path to test data file (txt, json, or jsonl)")
    parser.add_argument("--ground-truth", "-g", required=True,
                        help="Path to ground truth file (json or jsonl)")

    # Method selection arguments
    parser.add_argument("--methods", "-m", choices=["rag", "slm", "both"], default="both",
                        help="Which extraction methods to evaluate")

    # Output arguments
    parser.add_argument("--output", "-o", default="data/evaluation_results.csv",
                        help="Path to save evaluation results")
    parser.add_argument("--save-details", "-d", action="store_true",
                        help="Save detailed results")

    # Additional arguments
    parser.add_argument("--rag-query", default="Extract all property-value pairs from this document",
                        help="Query to use for RAG extraction")
    parser.add_argument("--slm-model", default=None,
                        help="Path to SLM model directory")
    parser.add_argument("--no-latency", action="store_true",
                        help="Don't measure extraction latency")

    args = parser.parse_args()

    # Load ground truth data
    print(f"Loading ground truth data from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded ground truth for {len(ground_truth)} documents")

    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_data = {}

    file_ext = os.path.splitext(args.test_data)[1].lower()
    if file_ext == '.jsonl':
        # Load from JSONL file
        with jsonlines.open(args.test_data, mode='r') as reader:
            for item in reader:
                if 'id' in item and 'text' in item:
                    test_data[item['id']] = item['text']
    elif file_ext in ['.json', '.txt']:
        # Load from JSON or TXT file
        text = load_document(args.test_data)
        # Use filename as document ID
        doc_id = os.path.basename(args.test_data).split('.')[0]
        test_data[doc_id] = text
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please provide a .json, .jsonl, or .txt file.")

    print(f"Loaded {len(test_data)} test documents")

    # Set up method arguments
    rag_args = {"query": args.rag_query} if args.rag_query else {}
    slm_args = {"model_dir": args.slm_model} if args.slm_model else {}

    # Measure latency unless disabled
    measure_latency = not args.no_latency

    # Run evaluation based on selected methods
    if args.methods == "rag":
        # Evaluate only RAG
        print("\nEvaluating RAG extraction method...")
        rag_results = evaluate_extraction(rag_extract, test_data, ground_truth, rag_args, measure_latency)

        # Create a DataFrame for the results
        results = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'Latency (s)'],
            'RAG': [
                rag_results['overall']['precision'],
                rag_results['overall']['recall'],
                rag_results['overall']['f1'],
                rag_results['overall']['latency']
            ]
        })

        # Store the detailed results
        results.attrs['rag_results'] = rag_results

    elif args.methods == "slm":
        # Evaluate only SLM
        print("\nEvaluating SLM extraction method...")
        slm_results = evaluate_extraction(slm_extract, test_data, ground_truth, slm_args, measure_latency)

        # Create a DataFrame for the results
        results = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1 Score', 'Latency (s)'],
            'SLM': [
                slm_results['overall']['precision'],
                slm_results['overall']['recall'],
                slm_results['overall']['f1'],
                slm_results['overall']['latency']
            ]
        })

        # Store the detailed results
        results.attrs['slm_results'] = slm_results

    else:  # args.methods == "both"
        # Compare both methods
        print("\nComparing RAG and SLM extraction methods...")
        results = compare_methods(test_data, ground_truth, rag_args, slm_args)

    # Print the results
    print("\nEvaluation Results:")
    print("------------------")
    print(results.to_string(index=False))

    # Log the results
    log_results(results, args.output, args.save_details)
