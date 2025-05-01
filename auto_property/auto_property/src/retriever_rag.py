"""
Retriever-Augmented Generation (RAG) Module for Auto Property

This module implements a LangChain-based RAG pipeline for extracting
property-value pairs from documents. It uses a vector database (FAISS)
to store document embeddings and a language model to generate answers.

Functions:
    load_documents: Loads preprocessed document chunks
    create_embeddings: Creates embeddings for document chunks
    setup_retriever: Sets up a LangChain retriever with FAISS
    extract_properties: Extracts property-value pairs using RAG
"""

import os
import json
import jsonlines
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Placeholder for actual implementation
def load_documents(jsonl_path: str) -> List[Document]:
    """
    Load preprocessed document chunks from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file containing document chunks

    Returns:
        List of LangChain Document objects
    """
    documents = []

    try:
        with jsonlines.open(jsonl_path, mode='r') as reader:
            for item in reader:
                if 'text' in item and 'id' in item:
                    doc = Document(
                        page_content=item['text'],
                        metadata={'id': item['id']}
                    )
                    documents.append(doc)
    except Exception as e:
        print(f"Error loading documents: {e}")

    return documents


def create_embeddings(embedding_type: str = "huggingface") -> Any:
    """
    Create embeddings using either OpenAI or HuggingFace models.

    Args:
        embedding_type: Type of embeddings to use ("openai" or "huggingface")

    Returns:
        Embedding model instance
    """
    if embedding_type.lower() == "openai":
        # Note: Requires OPENAI_API_KEY environment variable
        try:
            return OpenAIEmbeddings()
        except Exception as e:
            print(f"Error creating OpenAI embeddings: {e}")
            print("Falling back to HuggingFace embeddings")
            embedding_type = "huggingface"

    if embedding_type.lower() == "huggingface":
        # Use a sentence-transformers model that's good for embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            print(f"Error creating HuggingFace embeddings: {e}")
            raise

    raise ValueError(f"Unsupported embedding type: {embedding_type}")


def setup_retriever(documents: List[Document], embeddings: Any) -> Any:
    """
    Set up a LangChain retriever with FAISS vector store.

    Args:
        documents: List of LangChain Document objects
        embeddings: Embedding model instance

    Returns:
        LangChain retriever
    """
    if not documents:
        raise ValueError("No documents provided for indexing")

    try:
        # Create a FAISS vector store from the documents and embeddings
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={"k": 3}     # Return top 3 most similar documents
        )

        return retriever
    except Exception as e:
        print(f"Error setting up retriever: {e}")
        raise


def ask_question(retriever: Any, query: str, llm_type: str = "huggingface") -> str:
    """
    Use a retriever and LLM to answer a question.

    Args:
        retriever: LangChain retriever
        query: Question to ask
        llm_type: Type of LLM to use ("openai" or "huggingface")

    Returns:
        Answer from the LLM
    """
    # Create a prompt template for property extraction
    template = """
    You are an AI assistant specialized in extracting property-value pairs from text.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Format your response as a JSON object with property names as keys and their values as values.

    Context:
    {context}

    Question: {question}

    Answer (JSON format):
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Set up the language model
    if llm_type.lower() == "openai":
        # Note: Requires OPENAI_API_KEY environment variable
        try:
            llm = OpenAI(temperature=0)
        except Exception as e:
            print(f"Error creating OpenAI LLM: {e}")
            print("Falling back to HuggingFace LLM")
            llm_type = "huggingface"

    if llm_type.lower() == "huggingface":
        # Import HuggingFace LLM if needed
        from langchain.llms import HuggingFaceHub

        try:
            # Use a smaller model that's suitable for text generation
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature": 0.1, "max_length": 512}
            )
        except Exception as e:
            print(f"Error creating HuggingFace LLM: {e}")
            # Return a mock response for demonstration
            return """
            {
                "product": "Example Product",
                "price": "$29.99",
                "weight": "1.5 lbs",
                "dimensions": "10 x 5 x 2 inches",
                "note": "This is a mock response as LLM initialization failed"
            }
            """

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved documents into the prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Run the chain
    try:
        result = qa_chain.run(query)
        return result
    except Exception as e:
        print(f"Error running QA chain: {e}")
        # Return a mock response for demonstration
        return """
        {
            "error": "Failed to run QA chain",
            "mock_property": "mock_value",
            "note": "This is a mock response as QA chain execution failed"
        }
        """


def parse_llm_response(response: str) -> Dict[str, str]:
    """
    Parse the LLM's response into a dictionary of property-value pairs.

    Args:
        response: Response from the LLM

    Returns:
        Dictionary of property-value pairs
    """
    try:
        # Try to parse the response as JSON
        properties = json.loads(response)

        # Ensure all values are strings
        return {k: str(v) for k, v in properties.items()}
    except json.JSONDecodeError:
        # If the response is not valid JSON, try to extract key-value pairs using regex
        import re

        properties = {}
        # Look for patterns like "property": "value" or "property": value
        pattern = r'"([^"]+)":\s*"?([^",\}\n]+)"?'
        matches = re.findall(pattern, response)

        for prop, val in matches:
            properties[prop.strip()] = val.strip()

        if not properties:
            # If no properties were found, return an error message
            return {
                "error": "Failed to parse LLM response",
                "raw_response": response
            }

        return properties


def extract_properties(text: str, query: str = "Extract all property-value pairs from this document") -> Dict[str, str]:
    """
    Extract property-value pairs using RAG.

    Args:
        text: Text to extract properties from
        query: Query to send to the LLM

    Returns:
        Dictionary of property-value pairs
    """
    try:
        # Create a document from the text
        doc = Document(page_content=text)
        documents = [doc]

        # Create embeddings
        embeddings = create_embeddings("huggingface")

        # Set up retriever
        retriever = setup_retriever(documents, embeddings)

        # Ask the question
        response = ask_question(retriever, query, "huggingface")

        # Parse the response
        properties = parse_llm_response(response)

        return properties
    except Exception as e:
        print(f"Error in RAG extraction: {e}")

        # Return a fallback result for demonstration
        print("RAG extraction called with text:", text[:100] + "...")
        return {
            "rag_fallback": "Extraction failed, using fallback",
            "note": "To implement actual RAG, you would need to set up proper API keys and configurations"
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract property-value pairs using RAG")
    parser.add_argument("--input", "-i", help="Path to input file (JSONL or TXT)")
    parser.add_argument("--text", "-t", help="Text to extract properties from")
    parser.add_argument("--query", "-q", default="Extract all property-value pairs from this document",
                        help="Query to send to the LLM")
    parser.add_argument("--embedding", "-e", default="huggingface",
                        choices=["openai", "huggingface"],
                        help="Type of embeddings to use")
    parser.add_argument("--llm", "-l", default="huggingface",
                        choices=["openai", "huggingface"],
                        help="Type of LLM to use")

    args = parser.parse_args()

    # Check if either input file or text is provided
    if not args.input and not args.text:
        parser.error("Either --input or --text must be provided")

    # Get the text to extract properties from
    if args.input:
        # Check if the input file exists
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            exit(1)

        # Load the documents
        if args.input.endswith(".jsonl"):
            documents = load_documents(args.input)
            if not documents:
                print(f"Error: No documents found in {args.input}")
                exit(1)
            text = " ".join([doc.page_content for doc in documents])
        else:
            # Assume it's a text file
            with open(args.input, "r", encoding="utf-8") as f:
                text = f.read()
    else:
        text = args.text

    # Extract properties
    print(f"Extracting properties from text ({len(text)} characters)...")
    properties = extract_properties(text, args.query)

    # Print the properties
    print("\nExtracted Properties:")
    print("---------------------")
    for prop, value in properties.items():
        print(f"{prop}: {value}")

    print("\nDone!")
