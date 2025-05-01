#!/bin/bash

# Script to run and build the Auto Property project

# Function to display help
show_help() {
    echo "Usage: ./run_auto_property.sh [OPTION]"
    echo "Run and build the Auto Property project."
    echo ""
    echo "Options:"
    echo "  setup       Install dependencies and download NLTK data"
    echo "  preprocess  Run data preprocessing"
    echo "  rag         Run RAG-based property extraction"
    echo "  slm         Train and run SLM-based property extraction"
    echo "  test        Run tests comparing RAG and SLM approaches"
    echo "  app         Launch the Streamlit web interface"
    echo "  help        Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_auto_property.sh setup"
    echo "  ./run_auto_property.sh app"
}

# Function to setup the environment
setup() {
    echo "Setting up the Auto Property environment..."
    pip install -r auto_property/requirements.txt
    python download_nltk_data.py
    echo "Setup complete!"
}

# Function to run data preprocessing
preprocess() {
    echo "Running data preprocessing..."
    echo "Please provide the input file path (relative to the project root):"
    read input_file
    echo "Please provide the output JSONL file path (relative to the project root):"
    read output_jsonl
    echo "Please provide the output TXT file path for debugging (optional, press Enter to skip):"
    read output_txt

    if [ -z "$output_txt" ]; then
        python auto_property/src/data_preprocessing.py "$input_file" --output_jsonl "$output_jsonl"
    else
        python auto_property/src/data_preprocessing.py "$input_file" --output_jsonl "$output_jsonl" --output_txt "$output_txt"
    fi
}

# Function to run RAG-based property extraction
rag() {
    echo "Running RAG-based property extraction..."
    echo "Please choose an input option:"
    echo "1. Input file"
    echo "2. Input text"
    read option

    if [ "$option" == "1" ]; then
        echo "Please provide the input file path (relative to the project root):"
        read input_file
        echo "Do you want to customize the query? (y/n)"
        read customize_query

        if [ "$customize_query" == "y" ]; then
            echo "Please enter your query:"
            read query
            python auto_property/src/retriever_rag.py --input "$input_file" --query "$query"
        else
            python auto_property/src/retriever_rag.py --input "$input_file"
        fi
    else
        echo "Please enter the text to extract properties from:"
        read text
        python auto_property/src/retriever_rag.py --text "$text"
    fi
}

# Function to train and run SLM-based property extraction
slm() {
    echo "SLM-based property extraction..."
    echo "Please choose an option:"
    echo "1. Train a model"
    echo "2. Extract properties using a trained model"
    read option

    if [ "$option" == "1" ]; then
        echo "Please provide the input JSONL file path (relative to the project root):"
        read input_file
        echo "Please provide the output model directory (relative to the project root):"
        read output_dir
        python auto_property/src/slm_extractor.py train --input "$input_file" --output "$output_dir"
    else
        echo "Please choose an input option:"
        echo "1. Input file"
        echo "2. Input text"
        read input_option

        echo "Please provide the model directory (relative to the project root):"
        read model_dir

        if [ "$input_option" == "1" ]; then
            echo "Please provide the input file path (relative to the project root):"
            read input_file
            python auto_property/src/slm_extractor.py extract --input "$input_file" --model "$model_dir"
        else
            echo "Please enter the text to extract properties from:"
            read text
            python auto_property/src/slm_extractor.py extract --text "$text" --model "$model_dir"
        fi
    fi
}

# Function to run tests
test() {
    echo "Running tests comparing RAG and SLM approaches..."
    cd auto_property && python test_extraction.py && cd ..
}

# Function to launch the Streamlit app
app() {
    echo "Launching the Streamlit web interface..."
    streamlit run auto_property/src/app.py
}

# Main script logic
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    setup)
        setup
        ;;
    preprocess)
        preprocess
        ;;
    rag)
        rag
        ;;
    slm)
        slm
        ;;
    test)
        test
        ;;
    app)
        app
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac

exit 0
