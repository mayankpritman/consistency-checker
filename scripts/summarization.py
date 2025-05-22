# summarization.py
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI, AzureOpenAI
import pandas as pd

def normalize_text(text):
    """
    Normalize text by stripping extra spaces and standardizing case.
    """
    return ' '.join(text.strip().split()).lower()  # Lowercase and remove extra spaces
# Function to summarize the document using Azure OpenAI
def document_summary_azure(documents, api_key, azure_endpoint, azure_deployment):
    # documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    chunk_size = 1024
    documents = normalize_text(documents)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=100)
    nodes = splitter.split_text(documents)
    
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment
    )

    document_text = " ".join(nodes)
    prompt = f"""
    Summarize the document with the following details:
    1. Important Points: Provide the key points from the document as bullet points.
    2. Notable Sayings: Extract and highlight significant quotes or sayings from the document.
    3. Detailed Summary: Provide a comprehensive summary of the document, covering all main ideas, arguments, and conclusions in detail. Ensure the summary captures the essence of each section and provides a coherent overview of the entire document.

    **Word Limit**: Ensure the entire summary does not exceed 2000 words.

    Document text: {document_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=2048
    )
    return response.choices[0].message.content

# Function to summarize the document using OpenAI
def document_summary_openai(documents, api_key):
    # documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    documents = normalize_text(documents)
    chunk_size = 2048
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=100)
    nodes = splitter.split_text(documents)

    client = OpenAI(api_key=api_key)
    document_text = " ".join(nodes)
    prompt = f"""
    Summarize the document with the following details:
    1. Important Points: Provide the key points from the document as bullet points.
    2. Notable Sayings: Extract and highlight significant quotes or sayings from the document.
    3. Detailed Summary: Provide a comprehensive summary of the document, covering all main ideas, arguments, and conclusions in detail. Ensure the summary captures the essence of each section and provides a coherent overview of the entire document.

    **Word Limit**: Ensure the entire summary does not exceed 2000 words.

    Document text: {document_text}
    """

    response = client.completions.create(
        model="gpt-4", 
        prompt=prompt,
        max_tokens=2048,
        temperature=0.0
    )
    return response.choices[0].text.strip()

# Function to summarize a table using Azure OpenAI
def summarize_table_with_azure(table_df, api_key, azure_endpoint, azure_deployment):
    if isinstance(table_df, list):
        # Convert list of lists to DataFrame, assuming the first row is the header
        table_df = pd.DataFrame(table_df[1:], columns=table_df[0]) if table_df else pd.DataFrame()
    elif isinstance(table_df, pd.DataFrame):
        table_df = table_df
    else:
        raise ValueError("Unexpected table format")

    # Convert DataFrame to string
    table_text = table_df.to_string(index=False)
    prompt = f"""
    Summarize the following table:
    
    {table_text}

    Provide a concise summary highlighting the most important information.
    """

    # Initialize Azure OpenAI API client
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment
    )

    # Get summary from Azure OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=1024
    )
    return response.choices[0].message.content

# Function to summarize a table using OpenAI
def summarize_table_with_openai(table_df, api_key):
    table_text = table_df.to_string(index=False)  # Convert table dataframe to a string format
    prompt = f"""
    Summarize the following table:
    
    {table_text}

    Provide a concise summary highlighting the most important information.
    """

    client = OpenAI(api_key=api_key)

    # Get summary from OpenAI
    response = client.completions.create(
        model="gpt-4",  # Use the appropriate model name
        prompt=prompt,
        max_tokens=2048,
        temperature=0.0
    )
    return response.choices[0].text.strip()
