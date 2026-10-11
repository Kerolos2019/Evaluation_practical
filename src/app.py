"""Streamlit chatbot UI. Run via `streamlit run src/app.py` (or `python app.py`)."""


import sys
import uuid

from pathlib import Path


import streamlit as st


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import ask_with_tools
from src.design import PAGE_ICON, PAGE_TITLE, apply_design, render_bubble



st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")



apply_design()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"chatbot-{uuid.uuid4()}"


for message in st.session_state.messages:
    st.markdown(render_bubble(message["role"], message["content"]), unsafe_allow_html=True)


user_question = st.chat_input("Ask a question...")


if user_question:
 
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(render_bubble("user", user_question), unsafe_allow_html=True)


    with st.spinner("Thinking..."):

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        answer, _tool_names = ask_with_tools(
            user_question, chat_history=history, session_id=st.session_state.session_id
        )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(render_bubble("assistant", answer), unsafe_allow_html=True)
