# Changes Made to Address "Run and Build" Issue

## Summary
This document summarizes the changes made to address the "run and build" issue in the Auto Property project.

## Changes Made

1. **Updated Requirements**
   - Added NLTK back to the requirements.txt file, as it's used in data_preprocessing.py but was missing from the requirements.

2. **Created Helper Script**
   - Created a comprehensive shell script (`run_auto_property.sh`) that provides a user-friendly interface for running and building the Auto Property project.
   - The script includes the following functionality:
     - Setup: Install dependencies and download NLTK data
     - Preprocess: Run data preprocessing
     - RAG: Run RAG-based property extraction
     - SLM: Train and run SLM-based property extraction
     - Test: Run tests comparing RAG and SLM approaches
     - App: Launch the Streamlit web interface
     - Help: Display a help message
   - Made the script executable with `chmod +x run_auto_property.sh`

3. **Updated Documentation**
   - Updated the README.md file to include instructions for using the helper script
   - Added a new "Installation and Setup" section with instructions for both the helper script and manual installation
   - Updated the "Usage" section to mention that users can use either the helper script or run the commands directly

## How to Use

1. **Setup the Environment**
   ```bash
   ./run_auto_property.sh setup
   ```

2. **Run the Project**
   ```bash
   # Show all available options
   ./run_auto_property.sh help
   
   # Run a specific option
   ./run_auto_property.sh [option]  # where option is: preprocess, rag, slm, test, or app
   ```

## Final Status
The project now has a comprehensive helper script that simplifies the process of running and building the Auto Property project. The script provides a user-friendly interface for all the main functionality of the project, and the documentation has been updated to include instructions for using the script.