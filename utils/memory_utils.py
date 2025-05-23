# utils/memory_utils.py
import streamlit as st

def update_chat_memory(query, answer):
    st.session_state.history.append({
        "question": query,
        "answer": answer
    })
