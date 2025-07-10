import streamlit as st
import json
import re
import requests
from collections import defaultdict
from langchain.text_splitter import RecursiveJsonSplitter
import faiss
from sentence_transformers import SentenceTransformer

# Session state for chat history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# === OLLAMA + PHI3 ===
def query_phi3_ollama(prompt):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"].strip()

# === CACHED MODELS AND DATA ===
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_books(json_path, chunk_size=300):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    splitter = RecursiveJsonSplitter(max_chunk_size=chunk_size)
    return splitter.split_json(json_data=json_data, convert_lists=True)

@st.cache_data
def build_faiss_index(_datasource, _model):
    texts = []
    book_ids = []
    book_map = {}

    for doc in _datasource:
        book = list(doc.values())[0]
        book_id = book.get("title", "")
        book_map[book_id] = book
        book_ids.append(book_id)

        full_text = f"{book.get('title', '')} {book.get('author', '')} {book.get('summary', '')}"
        texts.append(full_text)

    embeddings = _model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, book_ids, book_map

def retrieve_top_ranked_book(query, model, index, book_ids, book_map):
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, k=1)
    if indices[0][0] < len(book_ids):
        best_book_id = book_ids[indices[0][0]]
        return book_map[best_book_id]
    return None

# === STREAMLIT UI ===
st.title("📚 Book Q&A App (Ollama + Phi3)")

query = st.text_input("Ask a question about the books:")

# Load resources
datasource = load_books("books.json")
embedding_model = load_embedding_model()
index, book_ids, book_map = build_faiss_index(datasource, embedding_model)

# Process question
if query:
    book = retrieve_top_ranked_book(query, embedding_model, index, book_ids, book_map)
    if book:
        prompt = (
            f"You are a knowledgeable librarian assistant. Your task is to answer questions based on the content of the following book:\n"
            f"---\n"
            f"Book Title: {book['title']}\n"
            f"Author: {book['author']}\n"
            f"Summary:\n{book['summary']}\n"
            f"---\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        with st.spinner("Asking Phi3..."):
            answer = query_phi3_ollama(prompt)

        # Update history
        st.session_state.qa_history.append({
            "book": f"{book['title']} by {book['author']}",
            "question": query,
            "answer": answer
        })

        # Display
        st.markdown(f"### Book: {book['title']} by {book['author']}")
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("No matching book found.")

# History section
if st.session_state.qa_history:
    st.markdown("---")
    st.markdown("## History")
    for entry in reversed(st.session_state.qa_history):
        st.markdown(f"**Q:** {entry['question']}")
        st.markdown(f"**A:** {entry['answer']}")
        st.markdown(f"_Book: {entry['book']}_")
        st.markdown("---")
