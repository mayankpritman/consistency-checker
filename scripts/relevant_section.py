from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained Sentence-BERT model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Split both documents into sentences (you can also chunk by paragraphs or sections)
doc1_sentences = ["Sentence 1 from Document 1", "Sentence 2 from Document 1", ...]
doc2_sentences = ["Sentence 1 from Document 2", "Sentence 2 from Document 2", ...]

# Encode sentences from both documents into embeddings
embeddings_doc1 = model.encode(doc1_sentences)
embeddings_doc2 = model.encode(doc2_sentences)

# Calculate similarity between each sentence from Document 1 and Document 2
similarity_scores = cosine_similarity(embeddings_doc1, embeddings_doc2)

# Extract the top N most similar sentence pairs (adjust threshold or N as needed)
top_similar_pairs = []
for i in range(len(similarity_scores)):
    for j in range(len(similarity_scores[i])):
        if similarity_scores[i][j] > 0.7:  # You can adjust this threshold based on your needs
            top_similar_pairs.append((doc1_sentences[i], doc2_sentences[j]))

# top_similar_pairs will contain sentence pairs from both documents with high semantic similarity
