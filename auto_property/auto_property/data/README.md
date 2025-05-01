# Data Directory

This directory contains raw and processed text/documents for the Auto Property project.

## Structure

- **Raw Data**: Place your raw text files or JSON documents here
- **Processed Data**: The system will store processed data here, including:
  - `clean_chunks.jsonl`: Cleaned and chunked text data
  - Debug text files (optional)

## Data Format

### Input Data
The system accepts the following input formats:
- `.txt` files containing raw text
- `.json` files with text content (various structures supported)

### Output Data
The system generates:
- `.jsonl` files where each line contains a JSON object with:
  ```json
  {
    "id": "chunk_001",
    "text": "cleaned and chunked text content..."
  }
  ```
- Optional `.txt` debug files showing the chunked content

## Usage

When running the data preprocessing module, you can specify input and output paths:

```bash
python src/data_preprocessing.py input.txt --output_jsonl data/clean_chunks.jsonl --output_txt data/debug_chunks.txt
```

Or use the default paths:

```bash
python src/data_preprocessing.py input.txt
```