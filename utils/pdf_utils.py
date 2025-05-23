# utils/pdf_utils.py
import re
from langchain_community.document_loaders import PyPDFLoader

def load_and_clean_pdf_from_path(path: str) -> str:
    loader = PyPDFLoader(path)
    docs = loader.load()
    text = " ".join([doc.page_content for doc in docs])
    return re.sub(r"/G([0-9A-Fa-f]{2})", lambda m: chr(int(m.group(1), 16)), text)
