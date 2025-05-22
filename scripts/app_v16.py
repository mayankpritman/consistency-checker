import streamlit as st
import fitz  # PyMuPDF
import pymupdf
import tempfile
import shutil
import os
import base64
from PIL import Image
from pdf_processing import extract_text_from_pdf, extract_tables_from_pdf, extract_text_from_docx, \
    extract_tables_from_docx, extract_text_from_txt
from summarization import document_summary_azure, document_summary_openai, summarize_table_with_azure, \
    summarize_table_with_openai
from api_config import save_api_details, load_api_details
from openai import OpenAI, AzureOpenAI
import random
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import csv
from api_config import save_api_details, load_api_details
from openai import OpenAI, AzureOpenAI
import os
import random


# Function to check if two files are identical (based on name, size, and content length)
def extract_random_part(text, part_length=100, start_index=None):
    """Extracts a random substring from the text, starting from a specific index"""
    if start_index is None:
        if len(text) <= part_length:
            return text
        start_index = random.randint(0, len(text) - part_length)
    return text[start_index:start_index + part_length]


# Function to check if two files are identical based on size, text length, and synchronized random part comparison
def are_files_identical(file1, file2, num_comparisons=3, part_length=100):
    # Check if file sizes are identical
    if file1.size != file2.size:
        return False  # Files with different sizes are not identical

    # Extract text content based on file typ
    if file1.name.endswith('.pdf') and file2.name.endswith('.pdf'):
        text1 = extract_text_from_pdf(file1)
        text2 = extract_text_from_pdf(file2)
    elif file1.name.endswith('.docx') and file2.name.endswith('.docx'):
        text1 = extract_text_from_docx(file1)
        text2 = extract_text_from_docx(file2)
    elif file1.name.endswith('.txt') and file2.name.endswith('.txt'):
        text1 = extract_text_from_txt(file1)
        text2 = extract_text_from_txt(file2)
    else:
        return False  # Unsupported file type

    # Check if the lengths of the extracted text are identical
    if len(text1) != len(text2):
        return False  # Files with different content lengths are not identical

    # Extract a random starting index for the comparison
    start_index = random.randint(0, min(len(text1), len(text2)) - part_length)

    # Compare multiple random parts from both files using the same start index
    for _ in range(num_comparisons):
        random_part1 = extract_random_part(text1, part_length, start_index)
        random_part2 = extract_random_part(text2, part_length, start_index)
        if random_part1 != random_part2:
            return False  # Return False if any comparison fails

    # If all checks pass, the files are considered identical
    return True


#### change ##########
def normalize_text(text):
    return ' '.join(text.strip().split()).lower()


def split_text_into_paragraphs(text):
    """
    Split the text into paragraphs using:
    1. `. \n` (a period followed by a newline)
    2. Keywords like 'Whereas', 'Article', 'Recital', etc.
    """
    # Split the text by `. \n` which indicates a paragraph end
    paragraph_split_keywords = r'(?<=\.)\s*\n'  # Match a period followed by newlines

    paragraphs = re.split(paragraph_split_keywords, text.strip())

    # Clean up paragraphs by stripping whitespace and removing any empty results
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Further clean-up: Remove any unnecessary references or footnotes
    cleaned_paragraphs = []
    for para in paragraphs:
        # Optionally, remove footnotes (like (1), (2), etc.)
        para = re.sub(r'\(\d+\)\s+[^.]+', '', para)
        cleaned_paragraphs.append(para.strip())

    return cleaned_paragraphs


def split_text_into_chunks(paragraphs, max_chunk_size=1000):
    """
    Combine paragraphs into chunks, ensuring each chunk has a maximum size of `max_chunk_size` words.
    If a chunk does not reach the size, add the next paragraph to the current chunk.
    """
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Add the paragraph to the current chunk
        if len(current_chunk.split()) + len(paragraph.split()) <= max_chunk_size:
            current_chunk += " " + paragraph
        else:
            # If the current chunk exceeds the limit, store it and start a new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph  # Start a new chunk with the current paragraph

    # Don't forget to add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Step 3: Load the paraphrase model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_embeddings(text_chunks):
    return model.encode(text_chunks)


# Step 4: Save similar chunks to CSV
def save_similar_chunks_to_csv(similar_pairs, csv_filename="similar_chunks_para.csv"):
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["pdf1_chunk", "pdf2_chunk", "similarity_score", "pdf1_chunk_number", "pdf2_chunk_number"])

        for i, (chunk1, chunk2, score) in enumerate(similar_pairs):
            chunk1_index = text_chunks_pdf1.index(chunk1) + 1  # +1 to make it 1-indexed
            chunk2_index = text_chunks_pdf2.index(chunk2) + 1  # +1 to make it 1-indexed

            writer.writerow([chunk1, chunk2, score, chunk1_index, chunk2_index])


# Step 5: Compute Cosine Similarity Between Chunks in Both Documents
def find_similar_chunks(embeddings1, embeddings2, text_chunks1, text_chunks2, threshold=0.5):
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    similar_pairs = []

    for i in range(similarity_matrix.shape[0]):
        matching_chunks = []
        for j in range(similarity_matrix.shape[1]):
            similarity_score = similarity_matrix[i, j]

            if similarity_score >= threshold:
                matching_chunks.append((text_chunks1[i], text_chunks2[j], similarity_score))

        if matching_chunks:
            similar_pairs.extend(matching_chunks)

    return similar_pairs


def create_comparison_prompt(chunk1, chunk2):
    prompt = f"""
    You are an AI assistant specializing in text analysis. Your task is to compare two provided chunks of text from two documents and identify contradictions, inconsistencies, and similarities and and give the exact text available in two text chunks without changing a single word from document1 and document2 for contradiction,inconsistency and similarity like Document 1 states: "Statement from document 1" and Document 2 states: "Statement from document 2" and then describe it and also don't give full stop after completing the statement from document1 and document2 and give structured output. Please follow these instructions:

    1. **Identify Contradictions**: Highlight any contradictions found between the two text chunks. Consider direct oppositions in key statements, data, or interpretations.

    2. **Identify Inconsistencies**: Identify any inconsistencies, including discrepancies in data, unclear statements, or information that does not align.

    3. **Identify Similarities**: Highlight significant similarities between the two text chunks. Look for common themes, repeated phrases, agreements in data, or aligned conclusions.

    4. **Categorization**: Organize your findings into clear categories: contradictions, inconsistencies, and similarities.

    **Text from Document 1:**
    {chunk1}

    **Text from Document 2:**
    {chunk2}

    **Summary of Contradictions, Inconsistencies, and Similarities:**
    """
    return prompt


# Step to analyze similarities and contradictions using GPT-4
def analyze_chunks_with_gpt4(chunk1, chunk2, api_details):
    prompt = create_comparison_prompt(chunk1, chunk2)
    if api_details["api_choice"] == "Azure OpenAI":
        # Call OpenAI API
        client = AzureOpenAI(
            api_key=api_details["api_key"],
            azure_endpoint=api_details["azure_endpoint"],
            api_version="2024-02-01",
            azure_deployment=api_details["azure_deployment"]
        )
    else:
        client = OpenAI(api_key=api_details["api_key"], )
    with st.spinner('Comparing documents...'):
        if api_details["api_choice"] == "Azure OpenAI":
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=1024
            )
            return response.choices[0].message.content
        else:
            response = client.completions.create(
                model="gpt-4",
                prompt=prompt, temperature=0, max_tokens=1024
            )
            return response.choices[0].text.strip()


# Function to compare documents and aggregate results
def compare_documents_and_aggregate(text_chunks_pdf1, text_chunks_pdf2, embeddings_pdf1, embeddings_pdf2, api_details):
    # Step 1: Check document lengths and decide comparison approach
    text_pdf1_length = len(' '.join(text_chunks_pdf1))
    text_pdf2_length = len(' '.join(text_chunks_pdf2))

    if text_pdf1_length < 1000 and text_pdf2_length < 1000:
        # Directly compare the full documents if both are below 2000 characters
        print("Documents are short. Direct comparison with GPT-4.")
        result = analyze_chunks_with_gpt4(' '.join(text_chunks_pdf1), ' '.join(text_chunks_pdf2), api_details)
        return result
    else:
        # Step 2: Find Similar Chunks if documents are larger than 2000 characters
        similar_pairs = find_similar_chunks(embeddings_pdf1, embeddings_pdf2, text_chunks_pdf1, text_chunks_pdf2,
                                            threshold=0.6)

        comparison_results = []

        # Step 3: Analyze each pair of similar chunks using GPT-4
        for chunk1, chunk2, score in similar_pairs:
            print(f"Analyzing similarity score: {score} between chunks...")
            result = analyze_chunks_with_gpt4(chunk1, chunk2, api_details)
            comparison_results.append(result)  # Save the comparison result

        # After collecting all individual results, send them to GPT-4 for further aggregation
        final_prompt = f"""
        You are an AI assistant specializing in document analysis. You have been provided with a series of comparison results between two documents. Your task is to analyze and categorize the following findings into contradictions, inconsistencies, and similarities and give the exact text available in two text chunks without changing a single word from document1 and document2 for contradiction,inconsistency and similarity like Document 1 states: "Statement from document 1" and Document 2 states: "Statement from document 2" and then describe it and give structured output :

        Please analyze the following responses and categorize them appropriately. Each response includes analysis from a chunk comparison between two documents. Your goal is to:

        1. Identify contradictions: Extract any findings that suggest direct oppositions between the documents.
        2. Identify inconsistencies: Extract any findings that suggest discrepancies or unclear statements.
        3. Identify similarities: Extract any findings that suggest agreements or common themes between the documents.
        Please don't give chunk number give standard output 
        Here are the comparison results:

        {''.join(comparison_results)}

        **Final Categorization of Contradictions, Inconsistencies, and Similarities:**
        """

        # Send the aggregated results to GPT-4 for final analysis
        if api_details["api_choice"] == "Azure OpenAI":
            # Call OpenAI API
            client = AzureOpenAI(
                api_key=api_details["api_key"],
                azure_endpoint=api_details["azure_endpoint"],
                api_version="2024-02-01",
                azure_deployment=api_details["azure_deployment"]
            )
        else:
            client = OpenAI(api_key=api_details["api_key"], )
        with st.spinner('Comparing documents...'):
            if api_details["api_choice"] == "Azure OpenAI":
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": final_prompt}],
                    temperature=0.0, max_tokens=1024
                )
                return response.choices[0].message.content
            else:
                response = client.completions.create(
                    model="gpt-4",
                    prompt=final_prompt, temperature=0, max_tokens=1024
                )
                return response.choices[0].text.strip()


# Function to read the PDF files and generate summaries
def get_document_summary(file, file_type, api_choice, api_key, azure_endpoint=None, azure_deployment=None):
    # Save the uploaded file temporarily
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())

    # Initialize summary and tables
    summary = ""
    table_summaries = []

    # Extract the document summary and tables based on file type
    if file_type == "PDF":
        summary = extract_text_from_pdf(file.name)
        tables = extract_tables_from_pdf(file.name)
    elif file_type == "DOCX":
        summary = extract_text_from_docx(file.name)
        tables = extract_tables_from_docx(file.name)
    elif file_type == "TXT":
        summary = extract_text_from_txt(file.name)
        tables = []  # No tables in TXT files

    # Summarize the extracted text using the selected API
    if api_choice == "Azure OpenAI":
        summary = document_summary_azure(summary, api_key, azure_endpoint, azure_deployment)
    elif api_choice == "OpenAI":
        summary = document_summary_openai(summary, api_key)

    # Summarize the extracted tables using the selected API
    for table in tables:
        if api_choice == "Azure OpenAI":
            table_summary = summarize_table_with_azure(table, api_key, azure_endpoint, azure_deployment)
        elif api_choice == "OpenAI":
            table_summary = summarize_table_with_openai(table, api_key)
        table_summaries.append(table_summary)

    return summary, table_summaries


# Function to input API details
def input_api_details():
    if 'api_choice' not in st.session_state:
        st.session_state['api_choice'] = None
    with st.form(key='api_details_form'):
        api_choice = st.selectbox("Choose API", options=["Azure OpenAI", "OpenAI"])

        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state['api_choice'] = api_choice
    if st.session_state['api_choice'] is not None:
        api_key = st.text_input(f"{st.session_state['api_choice']} API Key")
        azure_endpoint = st.text_input("Azure OpenAI Endpoint") if st.session_state[
                                                                       'api_choice'] == "Azure OpenAI" else None
        azure_deployment = st.text_input("Azure OpenAI Deployment") if st.session_state[
                                                                           'api_choice'] == "Azure OpenAI" else None

        submit_button = st.button(label='Submit')

        if submit_button:
            if api_key and (api_choice == "OpenAI" or (azure_endpoint and azure_deployment)):
                save_api_details(api_choice, api_key, azure_endpoint, azure_deployment)
                st.session_state.form_submitted = True
                st.success(f"{api_choice} details saved successfully")
            else:
                st.error("Please fill in all the required fields")


st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

# Place the button in the rightmost column
with col8:
    if st.button("🔑"):
        st.session_state.button_clicked = True

if st.session_state.button_clicked and not st.session_state.form_submitted:
    input_api_details()
api_details = load_api_details()

# Load and display the logo
# logo_path = "Capgemini_201x_logo.svg.png"  # Specify the path to your logo image
logo_path = "Invent_Logo.jpeg"
logo_image = Image.open(logo_path)
# st.title("🕵🏻 Consistency Checker")
new_width = 250  # Set the width you want
new_height = 150  # Set the height you want
logo_image = logo_image.resize((new_width, new_height))
# Create the header with the logo and title
col1, col2 = st.columns([1, 6])
col1.image(logo_image, use_column_width=True)
#col1.image(logo_image, use_column_width=False)
col2.markdown(
    """
    <div style="background-color: white; padding: 0; border-radius: 0; height: 120px; display: flex; align-items: center; justify-content: center; font-family: Arial, sans-serif;">
        <h1 style="color: black; text-align: center; margin: 0; padding: 0; font-size: 40px;">🕵🏻Consistency Checker</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Load API details from the config file
# api_details = load_api_details()

# Create two columns for side-by-side display of PDF uploaders
col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Upload the first File", type=["pdf", "docx", "txt"], key="pdf1")

with col2:
    file2 = st.file_uploader("Upload the second File", type=["pdf", "docx", "txt"], key="pdf2")

# checkbox_state = st.checkbox('Show Table Summary')

if file1 is not None:
    if file1.name.endswith('.pdf'):
        file1_type = "PDF"
    elif file1.name.endswith('.docx'):
        file1_type = "DOCX"
    elif file1.name.endswith('.txt'):
        file1_type = "TXT"

if file2 is not None:
    if file2.name.endswith('.pdf'):
        file2_type = "PDF"
    elif file2.name.endswith('.docx'):
        file2_type = "DOCX"
    elif file2.name.endswith('.txt'):
        file2_type = "TXT"

# Check if files are identical
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None


def extract_errors(response):
    """
    Extracts errors from the response related to Document 1 and Document 2.
    """
    doc1_errors = []
    doc2_errors = []

    # Regular expression patterns to match Document 1 and Document 2 errors
    doc1_pattern = r"Document 1 (states|mentions)(?::)? \"(.*?)\""
    doc2_pattern = r"Document 2 (states|mentions)(?::)? \"(.*?)\""

    # Find all matches for Document 1 and Document 2
    doc1_errors = re.findall(doc1_pattern, response)
    doc2_errors = re.findall(doc2_pattern, response)

    # Only the second part of the tuple contains the actual text
    doc1_errors = [error[1] for error in doc1_errors]
    doc2_errors = [error[1] for error in doc2_errors]

    return doc1_errors, doc2_errors


def highlight_errors_in_pdf(doc_errors, pdf_document, error_message, color):
    """
    Highlights the specified errors from a PDF document and adds annotations.
    """
    for page_nb in range(pdf_document.page_count):
        page = pdf_document[page_nb]
        for txt in doc_errors:
            rects = page.search_for(txt)
            if rects:
                # Highlight the found text
                highlight = page.add_highlight_annot(rects)
                highlight.set_colors(stroke=color)
                highlight.update()

                # Add a text annotation next to the highlighted text
                for inst in rects:
                    annot = page.add_text_annot((20, rects[0].y0), error_message, icon="Note")
                    annot.set_opacity(0.9)
                    annot.update()


def highlight_errors_in_pdf_doc2(doc2_errors, input_pdf_path_doc2, output_pdf_path_doc2, error_message_doc2, color):
    """
    Highlights the specified errors from Document 2 and saves the modified Document 2 PDF.
    """
    pdf_document = fitz.open(input_pdf_path_doc2)

    for page_nb in range(pdf_document.page_count):
        page = pdf_document[page_nb]
        for txt in doc2_errors:
            rects = page.search_for(txt)
            if rects:
                highlight = page.add_highlight_annot(rects)
                highlight.set_colors(stroke=color)
                highlight.update()

                for inst in rects:
                    annot = page.add_text_annot((20, rects[0].y0), error_message_doc2, icon="Note")
                    annot.set_opacity(0.9)
                    annot.update()

    pdf_document.save(output_pdf_path_doc2)


def save_pdf_with_highlights(input_pdf_doc1, input_pdf_doc2, comparison_results):
    """
    This function will extract the errors for contradictions, inconsistencies, and similarities from the comparison results,
    and highlight them in the corresponding PDFs with different colors.
    """
    # Extract the sections (contradictions, inconsistencies, similarities)
    contradictions = extract_section(comparison_results, "Contradictions")
    inconsistencies = extract_section(comparison_results, "Inconsistencies")
    similarities = extract_section(comparison_results, "Similarities")

    # Join the lists into strings before passing them to extract_errors
    contradictions_str = '\n'.join(contradictions)  # Join all the items in the list into a single string
    inconsistencies_str = '\n'.join(inconsistencies)
    similarities_str = '\n'.join(similarities)

    # Extract errors for Document 1 and Document 2 from each section
    doc1_contradictions, doc2_contradictions = extract_errors(contradictions_str)
    doc1_inconsistencies, doc2_inconsistencies = extract_errors(inconsistencies_str)
    doc1_similarities, doc2_similarities = extract_errors(similarities_str)

    # Define colors for each category
    contradiction_color = fitz.utils.getColor("red")
    inconsistency_color = fitz.utils.getColor("yellow")
    similarity_color = fitz.utils.getColor("green")

    # Save the PDFs with highlighted errors (single file for each document)
    with tempfile.NamedTemporaryFile(delete=False) as temp_doc1, tempfile.NamedTemporaryFile(delete=False) as temp_doc2:
        temp_doc1_path = temp_doc1.name
        temp_doc2_path = temp_doc2.name

        # Ensure temp files are closed before proceeding to write
        temp_doc1.close()
        temp_doc2.close()

        # Open Document 1 and Document 2 for editing
        doc1_pdf = fitz.open(input_pdf_doc1)
        doc2_pdf = fitz.open(input_pdf_doc2)

        # Highlight contradictions, inconsistencies, and similarities in Document 1
        highlight_errors_in_pdf(doc1_contradictions, doc1_pdf, "Contradiction in Document 1", contradiction_color)
        highlight_errors_in_pdf(doc1_inconsistencies, doc1_pdf, "Inconsistency in Document 1", inconsistency_color)
        highlight_errors_in_pdf(doc1_similarities, doc1_pdf, "Similarity in Document 1", similarity_color)

        # Highlight contradictions, inconsistencies, and similarities in Document 2
        highlight_errors_in_pdf(doc2_contradictions, doc2_pdf, "Contradiction in Document 2", contradiction_color)
        highlight_errors_in_pdf(doc2_inconsistencies, doc2_pdf, "Inconsistency in Document 2", inconsistency_color)
        highlight_errors_in_pdf(doc2_similarities, doc2_pdf, "Similarity in Document 2", similarity_color)

        # Save the modified PDFs
        doc1_pdf.save(temp_doc1_path)
        doc2_pdf.save(temp_doc2_path)

        # Return the paths to the modified files
        return temp_doc1_path, temp_doc2_path

# Check if file1 and file2 are uploaded
if file1 and file2:
    if are_files_identical(file1, file2):
        st.warning("The two files are identical. Please upload different documents.")
    else:
        if file1:
            if api_details:
                with st.spinner('Generating summary for File1...'):
                    text_summary1, table_summaries1 = get_document_summary(file1, file1_type, api_details["api_choice"],
                                                                           api_details["api_key"],
                                                                           api_details.get("azure_endpoint"),
                                                                           api_details.get("azure_deployment"))

                st.markdown(
                    "<h2 style='text-align: center; color: black; font-family: Arial, sans-serif;'>Summaries</h2>",
                    unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    # Text Summary Expander
                    with st.expander("Text Summary of First File"):
                        st.markdown(f"<p>{text_summary1.replace('#', '&#35;')}</p>", unsafe_allow_html=True)

                    # Table Summaries Expander
                    if table_summaries1:
                        with st.expander("Table Summaries of First File"):
                            for idx, table_summary in enumerate(table_summaries1, 1):
                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid #0071af; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                        <h5 style="color: #0071af; font-family: Arial, sans-serif; text-align: center;">Table {idx} Summary</h5>
                                        <p>{table_summary.replace('#', '&#35;')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.write("No table summary available")
            else:
                st.error("API details are not provided. Please input the details using the button above.")

        # Process for the second file
        if file2:
            if api_details:
                with st.spinner('Generating summary for File2...'):
                    text_summary2, table_summaries2 = get_document_summary(file2, file2_type, api_details["api_choice"],
                                                                           api_details["api_key"],
                                                                           api_details.get("azure_endpoint"),
                                                                           api_details.get("azure_deployment"))
                with col2:
                    # Text Summary Expander
                    with st.expander("Text Summary of Second File"):
                        st.markdown(f"<p>{text_summary2.replace('#', '&#35;')}</p>", unsafe_allow_html=True)

                    # Table Summaries Expander
                    if table_summaries2:
                        with st.expander("Table Summaries of Second File"):
                            for idx, table_summary in enumerate(table_summaries2, 1):
                                st.markdown(
                                    f"""
                                    <div style="border: 1px solid #0071af; padding: 10px; border-radius: 5px; margin-top: 10px;">
                                        <h5 style="color: #0071af; font-family: Arial, sans-serif; text-align: center;">Table {idx} Summary</h5>
                                        <p>{table_summary.replace('#', '&#35;')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.write("No table summary available")
            else:
                st.error("API details are not provided. Please input the details using the button above.")

        # Add comparison only if both files are uploaded
        if file1 and file2 and api_details:
            # Construct the prompt
            text_pdf1 = extract_text_from_pdf(file1)
            text_pdf2 = extract_text_from_pdf(file2)

            text_paragraphs_pdf1 = split_text_into_paragraphs(text_pdf1)
            text_paragraphs_pdf2 = split_text_into_paragraphs(text_pdf2)

            # Combine paragraphs into chunks that do not exceed 2000 words
            text_chunks_pdf1 = split_text_into_chunks(text_paragraphs_pdf1, max_chunk_size=2000)
            text_chunks_pdf2 = split_text_into_chunks(text_paragraphs_pdf2, max_chunk_size=2000)

            # Generate embeddings for both PDFs
            embeddings_pdf1 = get_embeddings(text_chunks_pdf1)
            embeddings_pdf2 = get_embeddings(text_chunks_pdf2)

            # Compare chunks and aggregate results
            comparison_results = compare_documents_and_aggregate(text_chunks_pdf1, text_chunks_pdf2,
                                                                 embeddings_pdf1,
                                                                 embeddings_pdf2, api_details)
            print(comparison_results)
            # Store the comparison result in session state so it's available across reruns
            # st.session_state.comparison_results = comparison_results

            contradictions = []
            inconsistencies = []
            similarities = []


            def extract_section(content, section_label):
                # First regex pattern for '###' headers (3 hash signs)
                section_pattern_1 = re.compile(rf"###\s{section_label}:(.*?)(?=\n###|\n\*|\Z)", re.DOTALL)
                match_1 = section_pattern_1.search(content)

                # Second regex pattern for '*' headers (1 asterisk)
                section_pattern_2 = re.compile(rf"\*{section_label}:(.*?)(?=\n\*\*|\n###|\Z)", re.DOTALL)
                match_2 = section_pattern_2.search(content)

                if match_1:
                    section_content = match_1.group(1).strip()
                elif match_2:
                    section_content = match_2.group(1).strip()
                else:
                    return []  # Return an empty list if no match is found

                # Split the section into individual items based on numbering or list patterns
                items = re.split(r"\d+\.\s", section_content)
                return [item.strip() for item in items if item.strip()]


            # Extracting content from all three categories
            contradictions = extract_section(comparison_results, "Contradictions")
            inconsistencies = extract_section(comparison_results, "Inconsistencies")
            similarities = extract_section(comparison_results, "Similarities")

            # Streamlit display setup
            modified_pdf1, modified_pdf2 = save_pdf_with_highlights(file1, file2, comparison_results)


            # Display the modified PDFs as download links

            def file_to_base64(file_path):
                with open(file_path, "rb") as file:
                    return base64.b64encode(file.read()).decode()


            # Encode your PDFs into base64
            pdf1_base64 = file_to_base64(modified_pdf1)
            pdf2_base64 = file_to_base64(modified_pdf2)
            # Streamlit display setup
            st.markdown(
                "<h2 style='text-align: center; color: black; font-family: Arial, sans-serif;'>Comparison Result</h2>",
                unsafe_allow_html=True)

            # Inject custom CSS for consistent expander size
            st.markdown("""
                <style>
                .streamlit-expanderHeader {
                    font-size: 18px;
                    font-weight: bold;
                    color: black;
                }
                .streamlit-expanderContent {
                    height: 150px;  /* Set this value to control the height */
                    overflow-y: auto;
                }
                /* Set the width of all expanders to be the same */
                .css-1v3fvcr { /* This targets the expander container */
                    width: 100% !important;
                }
                .streamlit-expander { /* Ensure each expander gets consistent width */
                    max-width: 100% !important;
                    width: 100% !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # Layout using columns for better control
            col1, col2, col3 = st.columns([1, 1, 1])  # Control column width with ratio (1, 3, 1)

            # Contradictions Expander
            with col1:
                with st.expander("Contradictions"):
                    if contradictions:
                        for item in contradictions:
                            st.markdown(f"**{item}**")  # Markdown for bold text
                    else:
                        st.write("No contradictions found.")

            # Inconsistencies Expander
            with col2:
                with st.expander("Inconsistencies"):
                    if inconsistencies:
                        for item in inconsistencies:
                            st.markdown(f"*{item}*")  # Markdown for italic text
                    else:
                        st.write("No inconsistencies found.")

            # Similarities Expander
            with col3:
                with st.expander("Similarities"):
                    if similarities:
                        for item in similarities:
                            st.markdown(f"**_ {item} _**")  # Markdown for bold and italic text
                    else:
                        st.write("No similarities found.")

            # Place the download buttons beside the sections
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown(
                    f"<a href='data:application/pdf;base64,{pdf1_base64}' download='highlighted_document_1.pdf'>"
                    "<button>Download Highlighted Document 1</button></a>", unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"<a href='data:application/pdf;base64,{pdf2_base64}' download='highlighted_document_2.pdf'>"
                    "<button>Download Highlighted Document 2</button></a>", unsafe_allow_html=True
                )
