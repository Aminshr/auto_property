Auto Property
A portfolio project for extracting property-value pairs from documents using both:

Retrieval-Augmented Generation (RAG) with LangChain
Fine-tuned Small Language Models (SLMs) like DistilBERT or FLAN-T5
Project Structure
auto_property/
├── data/            # Raw and processed text/documents
├── models/          # Fine-tuned small language models
├── notebooks/       # Jupyter notebooks for experimentation
├── src/
│   ├── data_preprocessing.py  # Functions for text cleaning, chunking, label creation
│   ├── retriever_rag.py       # LangChain pipeline: FAISS + LLM for RAG extraction
│   ├── slm_extractor.py       # Fine-tune SLM to extract property-value pairs
│   ├── evaluation.py          # Compare accuracy of RAG vs SLM
│   └── app.py                 # Streamlit app: upload doc → choose model → view results
├── requirements.txt  # Dependencies
├── README.md         # Documentation
└── .gitignore        # Ignored files
Features
Text preprocessing and chunking
RAG-based property extraction using LangChain
Fine-tuned SLM for property-value extraction
Evaluation metrics comparing both approaches
Streamlit web interface for demonstration
Installation and Setup
Using the Helper Script
We provide a helper script to simplify installation and running the project:

# Make the script executable (if not already)
chmod +x run_auto_property.sh

# Setup the environment (install dependencies and download NLTK data)
./run_auto_property.sh setup

# Show all available options
./run_auto_property.sh help
Manual Installation
If you prefer to install manually:

# From the project root directory
pip install -r auto_property/requirements.txt
python download_nltk_data.py
Usage
You can use either the helper script or run the commands directly:

# Using the helper script
./run_auto_property.sh [option]  # where option is: preprocess, rag, slm, test, or app
Data Preprocessing
Process raw documents into cleaned, chunked text:

# From the project root directory
# Process a text file
python auto_property/src/data_preprocessing.py data/sample_input.txt --output_jsonl data/clean_chunks.jsonl --output_txt data/debug_chunks.txt

# Process a JSON file
python auto_property/src/data_preprocessing.py data/sample_input.json --output_jsonl data/json_clean_chunks.jsonl
RAG-based Property Extraction
Extract property-value pairs using Retrieval-Augmented Generation:

# From the project root directory
# Extract from a text file
python auto_property/src/retriever_rag.py --input data/sample_input.txt

# Extract from a JSONL file
python auto_property/src/retriever_rag.py --input data/clean_chunks.jsonl

# Extract from text directly
python auto_property/src/retriever_rag.py --text "This product weighs 1.5 lbs and costs $29.99"

# Customize the query
python auto_property/src/retriever_rag.py --input data/sample_input.txt --query "What is the weight and price of this product?"

SLM-based Property Extraction
Train and use a Small Language Model for property extraction:

# From the project root directory
# Train a model
python auto_property/src/slm_extractor.py train --input data/clean_chunks.jsonl --output models/property_extractor

# Extract properties using a trained model
python auto_property/src/slm_extractor.py extract --input data/sample_input.txt --model models/property_extractor

# Extract properties from text directly
python auto_property/src/slm_extractor.py extract --text "This product weighs 1.5 lbs and costs $29.99"

Testing Both Approaches
Run the test script to compare RAG and SLM approaches:

# From the project root directory
python auto_property/test_extraction.py
# Or use the helper script
./run_auto_property.sh test
Streamlit Web Interface
Launch the Streamlit app for interactive property extraction:

# From the project root directory
streamlit run auto_property/src/app.py
# Or use the helper script
./run_auto_property.sh app
The app allows you to:

Upload a document (TXT or JSON)
Choose between RAG and SLM extraction methods
View the extracted property-value pairs
Download the results as CSV
Notes on Implementation
RAG Implementation: Uses LangChain with FAISS for vector storage and retrieval. Can use either OpenAI or HuggingFace embeddings and LLMs.
SLM Implementation: Uses HuggingFace Transformers for token classification. Trains a model to identify property and value tokens.
Fallback Mechanisms: Both approaches include pattern matching as a fallback when model-based extraction fails.
API Keys: To use OpenAI models, set the OPENAI_API_KEY environment variable. To use HuggingFace Hub, set the HUGGINGFACE_API_KEY environment variable.
