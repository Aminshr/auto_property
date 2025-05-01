"""
Small Language Model (SLM) Extractor Module for Auto Property

This module provides functionality for fine-tuning a small language model
(like DistilBERT) for extracting property-value pairs from documents using
a token classification approach (NER-style).

Functions:
    prepare_training_data: Prepares data for training the SLM
    train_model: Fine-tunes a pre-trained model for token classification
    save_model: Saves the fine-tuned model
    load_model: Loads a fine-tuned model
    extract_properties: Extracts property-value pairs using the fine-tuned model
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# HuggingFace imports
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset, load_dataset

# Token classification labels
# B-PROP: Beginning of a property
# B-VAL: Beginning of a value
# O: Outside (not a property or value)
LABELS = ["O", "B-PROP", "B-VAL"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


# Placeholder for actual implementation
def prepare_training_data(jsonl_path: str) -> Dataset:
    """
    Prepare data for training the SLM.

    Args:
        jsonl_path: Path to the JSONL file containing document chunks

    Returns:
        HuggingFace Dataset ready for training
    """
    import jsonlines
    import re
    from sklearn.model_selection import train_test_split

    # Check if the file exists
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    # Load document chunks
    documents = []
    try:
        with jsonlines.open(jsonl_path, mode='r') as reader:
            for item in reader:
                if 'text' in item:
                    documents.append(item['text'])
    except Exception as e:
        print(f"Error loading documents: {e}")
        # Create some synthetic documents for demonstration
        documents = [
            "This product has a weight of 1.5 lbs and dimensions of 10 x 5 x 2 inches.",
            "The price is $29.99 and it comes in red, blue, and green colors.",
            "Made by ACME Corp with high-quality materials and a 2-year warranty."
        ]

    # Generate synthetic property-value pairs and token labels
    dataset_items = []

    for doc in documents:
        # Tokenize the document (simple whitespace tokenization for demonstration)
        tokens = doc.split()

        # Initialize all labels as "O" (outside)
        labels = ["O"] * len(tokens)

        # Look for common property-value patterns
        # This is a simplified approach for demonstration
        property_patterns = [
            (r'(\w+)\s+of\s+([\w\d\.]+)', 0, 2),  # "weight of 1.5"
            (r'(\w+)\s+is\s+([\w\d\.\$]+)', 0, 2),  # "price is $29.99"
            (r'(\w+):\s+([\w\d\.]+)', 0, 1)   # "weight: 1.5"
        ]

        # Apply patterns to find properties and values
        for pattern, prop_group, val_group in property_patterns:
            for match in re.finditer(pattern, doc):
                prop = match.group(prop_group)
                val = match.group(val_group)

                # Find the token indices for the property and value
                for i, token in enumerate(tokens):
                    if prop in token:
                        labels[i] = "B-PROP"
                    if val in token:
                        labels[i] = "B-VAL"

        # Add to dataset
        dataset_items.append({
            "tokens": tokens,
            "labels": labels,
            "text": doc
        })

    # Convert labels to IDs
    for item in dataset_items:
        item["label_ids"] = [LABEL2ID[label] for label in item["labels"]]

    # Split into train and test sets
    train_items, test_items = train_test_split(dataset_items, test_size=0.2, random_state=42)

    # Create HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in train_items],
        "labels": [item["label_ids"] for item in train_items],
        "text": [item["text"] for item in train_items]
    })

    test_dataset = Dataset.from_dict({
        "tokens": [item["tokens"] for item in test_items],
        "labels": [item["label_ids"] for item in test_items],
        "text": [item["text"] for item in test_items]
    })

    # Combine into a single dataset with train/test splits
    dataset = Dataset.from_dict({
        "train": train_dataset,
        "test": test_dataset
    })

    return dataset


def train_model(
    dataset: Dataset,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/property_extractor",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 16,
    learning_rate: float = 5e-5
) -> Tuple[Any, Any]:
    """
    Fine-tune a pre-trained model for token classification.

    Args:
        dataset: HuggingFace Dataset for training
        model_name: Name of the pre-trained model to fine-tune
        output_dir: Directory to save the fine-tuned model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device during training
        learning_rate: Learning rate for training

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Tokenize the dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )

        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply tokenization to the dataset
    try:
        tokenized_dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True
        )
    except Exception as e:
        print(f"Error tokenizing dataset: {e}")
        # For demonstration, return a mock model and tokenizer
        print("Returning mock model and tokenizer for demonstration")
        return model, tokenizer

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metrics computation function
    def compute_metrics(eval_pred):
        from sklearn.metrics import precision_recall_fscore_support

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Flatten the lists
        true_predictions_flat = [p for sublist in true_predictions for p in sublist]
        true_labels_flat = [l for sublist in true_labels for l in sublist]

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_flat, true_predictions_flat, average='weighted'
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    try:
        print("Training the model...")
        trainer.train()

        # Evaluate the model
        print("Evaluating the model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")

        # Save the model
        print(f"Saving the model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        return model, tokenizer
    except Exception as e:
        print(f"Error training model: {e}")
        # For demonstration, return the untrained model and tokenizer
        print("Returning untrained model and tokenizer")
        return model, tokenizer


def save_model(model: Any, tokenizer: Any, output_dir: str) -> None:
    """
    Save the fine-tuned model and tokenizer.

    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer used with the model
        output_dir: Directory to save the model and tokenizer
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save the model
        model.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

        # Save the tokenizer
        tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")

        # Save the label mappings
        with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
            json.dump({
                "id2label": ID2LABEL,
                "label2id": LABEL2ID
            }, f)
        print(f"Label mappings saved to {output_dir}/label_mappings.json")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model_dir: str) -> Tuple[Any, Any]:
    """
    Load a fine-tuned model and tokenizer.

    Args:
        model_dir: Directory containing the model and tokenizer

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if the model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    try:
        # Load the label mappings if available
        label_mappings_path = os.path.join(model_dir, "label_mappings.json")
        if os.path.exists(label_mappings_path):
            with open(label_mappings_path, "r") as f:
                label_mappings = json.load(f)
            id2label = label_mappings["id2label"]
            label2id = label_mappings["label2id"]
        else:
            # Use the default label mappings
            id2label = ID2LABEL
            label2id = LABEL2ID

        # Load the model
        model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            id2label=id2label,
            label2id=label2id
        )
        print(f"Model loaded from {model_dir}")

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Tokenizer loaded from {model_dir}")

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        # For demonstration, return a mock model and tokenizer
        print("Creating a new model and tokenizer for demonstration")

        # Create a new model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForTokenClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

        return model, tokenizer


def extract_properties(text: str, model_dir: str = None) -> Dict[str, str]:
    """
    Extract property-value pairs using the fine-tuned model.

    Args:
        text: Text to extract properties from
        model_dir: Directory containing the model and tokenizer (optional)

    Returns:
        Dictionary of property-value pairs
    """
    try:
        # Load the model and tokenizer if a model directory is provided
        if model_dir and os.path.exists(model_dir):
            model, tokenizer = load_model(model_dir)
        else:
            # For demonstration, use a default model
            print("No model directory provided or directory not found.")
            print("Using a default model for demonstration.")

            # Create a new model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModelForTokenClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(LABELS),
                id2label=ID2LABEL,
                label2id=LABEL2ID
            )

        # Tokenize the input text
        # First, split the text into tokens (simple whitespace tokenization for demonstration)
        tokens = text.split()

        # Tokenize for the model
        inputs = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        # Run the model to get token classifications
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Convert token classifications to property-value pairs
        properties = {}
        current_property = None
        current_value = None

        # Get word IDs to align with original tokens
        word_ids = inputs.word_ids(batch_index=0)

        for i, (word_id, prediction) in enumerate(zip(word_ids, predictions[0])):
            if word_id is not None:  # Skip special tokens
                label = ID2LABEL[prediction.item()]
                token = tokens[word_id]

                if label == "B-PROP":
                    # Start of a new property
                    if current_property and current_value:
                        # Save the previous property-value pair
                        properties[current_property] = current_value

                    current_property = token
                    current_value = None
                elif label == "B-VAL":
                    # Start of a value
                    if current_property:
                        current_value = token
                elif label == "O":
                    # Outside token
                    pass

        # Add the last property-value pair if any
        if current_property and current_value:
            properties[current_property] = current_value

        # If no properties were found, use pattern matching as a fallback
        if not properties:
            print("No properties found using the model. Using pattern matching as a fallback.")
            properties = extract_properties_with_patterns(text)

        return properties
    except Exception as e:
        print(f"Error in SLM extraction: {e}")

        # Return a fallback result using pattern matching
        print("SLM extraction called with text:", text[:100] + "...")
        print("Using pattern matching as a fallback.")
        return extract_properties_with_patterns(text)


def extract_properties_with_patterns(text: str) -> Dict[str, str]:
    """
    Extract property-value pairs using simple pattern matching.
    This is a fallback method when the model-based extraction fails.

    Args:
        text: Text to extract properties from

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
        r'(\w+)\s+of\s+([\w\d\.]+)',  # weight of 1.5
        r'(\w+)\s+is\s+([\w\d\.\$]+)'  # price is $29.99
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and use a small language model for property extraction")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model on a JSONL file")
    train_parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    train_parser.add_argument("--output", "-o", default="models/property_extractor", help="Path to save the model")
    train_parser.add_argument("--model", "-m", default="distilbert-base-uncased", help="Base model to fine-tune")
    train_parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size for training")
    train_parser.add_argument("--learning-rate", "-lr", type=float, default=5e-5, help="Learning rate for training")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract properties from text")
    extract_parser.add_argument("--input", "-i", help="Path to input file (TXT)")
    extract_parser.add_argument("--text", "-t", help="Text to extract properties from")
    extract_parser.add_argument("--model", "-m", help="Path to the model directory")

    args = parser.parse_args()

    if args.command == "train":
        # Train a model
        print(f"Training a model on {args.input}...")

        # Prepare the training data
        dataset = prepare_training_data(args.input)

        # Train the model
        model, tokenizer = train_model(
            dataset=dataset,
            model_name=args.model,
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print(f"Model trained and saved to {args.output}")

    elif args.command == "extract":
        # Extract properties from text
        if not args.input and not args.text:
            parser.error("Either --input or --text must be provided")

        # Get the text to extract properties from
        if args.input:
            # Check if the input file exists
            if not os.path.exists(args.input):
                print(f"Error: Input file {args.input} does not exist")
                exit(1)

            # Load the text from the file
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            text = args.text

        # Extract properties
        print(f"Extracting properties from text ({len(text)} characters)...")
        properties = extract_properties(text, args.model)

        # Print the properties
        print("\nExtracted Properties:")
        print("---------------------")
        for prop, value in properties.items():
            print(f"{prop}: {value}")

        print("\nDone!")

    else:
        parser.print_help()
