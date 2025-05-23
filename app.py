# import streamlit as st
# from utils.embed_utils import load_embed_model
# from utils.faiss_utils import load_or_create_index, search_query
# from utils.memory_utils import update_chat_memory
# from transformers import pipeline

# PDF_PATH = "../NBC 2016-VOL.1.pdf-200-225.pdf"

# st.set_page_config(page_title="PDF Chatbot with Memory", page_icon="💬", layout="wide")
# st.title("💬 PDF Chatbot")

# if "history" not in st.session_state:
#     st.session_state.history = []

# # Load model and index once (cached)
# model = load_embed_model()
# with st.spinner("📦 Loading FAISS index and PDF chunks..."):
#     chunks, index = load_or_create_index(PDF_PATH, model)
#     qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# st.success("✅ Ready to chat with your PDF!")

# query = st.chat_input("Ask something about the PDF...")

# # Show chat history
# for entry in st.session_state.history:
#     with st.chat_message("user"):
#         st.markdown(entry["question"])
#     with st.chat_message("assistant"):
#         st.code(entry["answer"], language="markdown")
#         # Copy button + simple toast feedback
#         if st.button("Copy Answer", key=f"copy_{entry['question'][:10]}"):
#             st.experimental_set_query_params()  # dummy to refresh UI
#             st.toast("Copied to clipboard!")

# if query:
#     with st.chat_message("user"):
#         st.markdown(query)

#     with st.spinner("🤖 Searching..."):
#         results = search_query(query, index, model, chunks)
#         answer = results[0][0].page_content if results else "Sorry, I couldn't find anything relevant."

#     with st.chat_message("assistant"):
#         st.code(answer, language="markdown")
#         if st.button("Copy Answer", key="copy_latest"):
#             st.toast("Copied to clipboard!")

#     update_chat_memory(query, answer)



import streamlit as st
from transformers import pipeline
from utils.embed_utils import load_embed_model
from utils.faiss_utils import load_or_create_index, search_query
from utils.memory_utils import update_chat_memory

PDF_PATH = "../NBC 2016-VOL.1.pdf-200-225.pdf"

st.set_page_config(page_title="PDF Chatbot with Memory", page_icon="💬", layout="wide")
st.title("💬 PDF Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

# Load embedding model & FAISS index
model = load_embed_model()
with st.spinner("📦 Loading FAISS index and PDF chunks..."):
    chunks, index = load_or_create_index(PDF_PATH, model)

# Load QA pipeline from Hugging Face
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Display chat history
for i, entry in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.code(entry["answer"], language="markdown")
        if st.button("Copy Answer", key=f"copy_{i}"):
            st.toast("Copied to clipboard!")

# Take user input
query = st.chat_input("Ask something about the PDF...")

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 Thinking..."):
        # Get relevant chunks from FAISS
        top_k = 3
        results = search_query(query, index, model, chunks)
        context = " ".join([doc.page_content for doc, _ in results])

        # Run QA model on combined context
        try:
            response = qa_pipeline(question=query, context=context)
            answer = response["answer"]
        except Exception as e:
            answer = "Error answering the question. Try again."

    with st.chat_message("assistant"):
        st.code(answer, language="markdown")
        if st.button("Copy Answer", key="copy_latest"):
            st.toast("Copied to clipboard!")

    update_chat_memory(query, answer)
