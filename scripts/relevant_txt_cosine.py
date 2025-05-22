import fitz  # PyMuPDF

#def extract_text_from_pdf(pdf_path):
 #   doc = fitz.open(pdf_path)
#    text = ""
#    for page in doc:
#        text += page.get_text("text")
#    return text
#/*

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(text_chunks):
    embeddings = model.encode(text_chunks)
    return embeddings

# Example text chunks from extracted PDFs
text_chunks_pdf1 = ["chunk1 from PDF1", "chunk2 from PDF1"]
text_chunks_pdf2 = ["chunk1 from PDF2", "chunk2 from PDF2"]

embeddings_pdf1 = get_embeddings(text_chunks_pdf1)
embeddings_pdf2 = get_embeddings(text_chunks_pdf2)