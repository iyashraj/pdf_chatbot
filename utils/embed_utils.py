# utils/embed_utils.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_embed_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def embed_chunks(chunks, model):
    texts = [doc.page_content for doc in chunks]
    return model.encode(texts, show_progress_bar=True)

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents([Document(page_content=text)])
