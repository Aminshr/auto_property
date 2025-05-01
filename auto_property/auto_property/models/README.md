# Models Directory

This directory stores fine-tuned Small Language Models (SLMs) for the Auto Property project.

## Purpose

The models stored here are used for extracting property-value pairs from documents. These are typically fine-tuned versions of pre-trained models like:
- DistilBERT
- FLAN-T5
- Other HuggingFace transformer models

## Model Format

Models are saved in the HuggingFace format, which includes:
- Model weights (`.bin`, `.pt`, or `.pth` files)
- Model configuration (`.json` files)
- Tokenizer files

## Usage

When training a model using the `slm_extractor.py` module, you can specify the output directory:

```bash
python src/slm_extractor.py --output_dir models/my_property_extractor
```

To load a trained model:

```python
from src.slm_extractor import load_model

model, tokenizer = load_model("models/my_property_extractor")
```

## Model Selection

The project supports comparing different model architectures and sizes. When selecting a model, consider:
- Size vs. performance tradeoffs
- Inference speed requirements
- Accuracy needs

Smaller models like DistilBERT are faster but may be less accurate than larger models.

## Note

Large model files are excluded from Git tracking via the `.gitignore` file. If sharing this project, you may need to provide instructions for downloading or training the models.