# utils/faiss_utils.py
import os
import faiss
import pickle
import numpy as np
from utils.pdf_utils import load_and_clean_pdf_from_path
from utils.embed_utils import split_text, embed_chunks

INDEX_PATH = "data/faiss.index"
CHUNK_PATH = "data/chunks.pkl"

def save_index(index, path): faiss.write_index(index, path)
def load_index(path): return faiss.read_index(path)
def save_chunks(chunks, path): pickle.dump(chunks, open(path, "wb"))
def load_chunks(path): return pickle.load(open(path, "rb"))

def build_index_and_chunks(pdf_path, model):
    text = load_and_clean_pdf_from_path(pdf_path)
    chunks = split_text(text)
    embeddings = embed_chunks(chunks, model)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    save_index(index, INDEX_PATH)
    save_chunks(chunks, CHUNK_PATH)
    return chunks, index

# def load_or_create_index(pdf_path, model):
#     if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_PATH):
#         return load_chunks(CHUNK_PATH), load_index(INDEX_PATH)
#     return build_index_and_chunks(pdf_path, model)

def load_or_create_index(pdf_path, model):
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNK_PATH):
            if os.path.getsize(CHUNK_PATH) > 0 and os.path.getsize(INDEX_PATH) > 0:
                return load_chunks(CHUNK_PATH), load_index(INDEX_PATH)
            else:
                print("⚠️ One or both cached files are empty. Rebuilding index and chunks.")
    except Exception as e:
        print(f"⚠️ Error loading index or chunks: {e}. Rebuilding...")

    return build_index_and_chunks(pdf_path, model)


def search_query(query, index, model, chunks, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [(chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
