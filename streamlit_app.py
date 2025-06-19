import asyncio
import json
import os
import textwrap
from typing import List, Union

import httpx
import streamlit as st  # type: ignore

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BACKEND_URL}/query/stream"

st.set_page_config(page_title="SoloRAG Chat", page_icon="🤖")
st.title("SoloRAG – Stripe FAQ Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []  # type: ignore

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:  # assistant
        with st.chat_message("assistant"):
            st.markdown(msg["answer"])
            with st.expander("Show Sources"):
                st.markdown(msg["sources"])

# --- user input form ---
if prompt := st.chat_input("Ask a question about Stripe payments…"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        answer_box = st.empty()
        sources_container = st.container()

    async def fetch_stream(question: str):
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", QUERY_ENDPOINT, json={"question": question}) as r:
                buffer = ""
                async for chunk in r.aiter_text():
                    buffer += chunk
                    if "[SOURCES]" in buffer:
                        body, src = buffer.split("[SOURCES]", 1)
                        answer_box.markdown(body)
                        formatted_raw = src
                        try:
                            src_json = json.loads(src)
                            formatted_raw = "\n".join(
                                f"* {textwrap.shorten(s['text'], 120)} (score: {s['score']:.2f})" for s in src_json
                            )
                        except Exception:
                            pass

                        with sources_container.expander("Show Sources"):
                            st.markdown(formatted_raw)
                        return body, formatted_raw
                    answer_box.markdown(buffer + " ▌")

            # Fallback (should not hit)
        return "", ""

    # Fetch answer asynchronously
    body, formatted = asyncio.run(fetch_stream(prompt))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": body,
            "sources": formatted,
        }
    ) 