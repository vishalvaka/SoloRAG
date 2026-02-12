"""SoloRAG Streamlit Frontend -- with login, chat history, favorites, and preferences."""

import asyncio
import json
import os
import textwrap
import uuid
from typing import Union

import httpx
import streamlit as st  # type: ignore

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SoloRAG Chat", page_icon="🤖", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════════

# Restore token from query params so login survives page refreshes
_qp = st.query_params
if "token" not in st.session_state:
    st.session_state.token = _qp.get("token", None)
if "username" not in st.session_state:
    st.session_state.username = _qp.get("username", None)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = uuid.uuid4().hex[:16]
if "favorited" not in st.session_state:
    st.session_state.favorited = set()  # indices of messages already favorited


def _persist_auth(token: str, username: str):
    """Save auth to session state and query params (survives refresh)."""
    st.session_state.token = token
    st.session_state.username = username
    st.query_params["token"] = token
    st.query_params["username"] = username


def _clear_auth():
    """Clear auth from session state and query params."""
    st.session_state.token = None
    st.session_state.username = None
    if "token" in st.query_params:
        del st.query_params["token"]
    if "username" in st.query_params:
        del st.query_params["username"]


# Validate restored token on load -- if the backend rejects it, clear auth
if st.session_state.token and "token_validated" not in st.session_state:
    try:
        _check = httpx.get(f"{BACKEND_URL}/preferences", headers=_auth_headers(), timeout=5)
        if _check.status_code == 401:
            _clear_auth()
    except Exception:
        pass  # backend may be down; keep token and let user retry
    st.session_state.token_validated = True


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}


# ═══════════════════════════════════════════════════════════════════════════
# Login / Register page
# ═══════════════════════════════════════════════════════════════════════════

def show_auth_page():
    st.title("SoloRAG -- Login")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted and username and password:
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/auth/login",
                    json={"username": username, "password": password},
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    _persist_auth(data["token"], data["username"])
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Login failed"))
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password (min 6 chars)", type="password")
            reg_submitted = st.form_submit_button("Register", type="primary")
        if reg_submitted and new_user and new_pass:
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/auth/register",
                    json={"username": new_user, "password": new_pass},
                    timeout=10,
                )
                if resp.status_code == 201:
                    data = resp.json()
                    _persist_auth(data["token"], data["username"])
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Registration failed"))
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar (user features)
# ═══════════════════════════════════════════════════════════════════════════

def show_sidebar():
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Logout"):
            # Try to invalidate server-side session
            try:
                httpx.post(f"{BACKEND_URL}/auth/logout", headers=_auth_headers(), timeout=5)
            except Exception:
                pass
            _clear_auth()
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # New conversation
        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = uuid.uuid4().hex[:16]
            st.session_state.favorited = set()
            st.rerun()

        # Chat history
        st.subheader("Chat History")
        try:
            resp = httpx.get(f"{BACKEND_URL}/chat/history", headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                convos = resp.json()
                for c in convos[:10]:
                    label = f"{c['conversation_id'][:8]}... ({c['message_count']} msgs)"
                    if st.button(label, key=f"conv_{c['conversation_id']}"):
                        # Load conversation
                        r2 = httpx.get(
                            f"{BACKEND_URL}/chat/history/{c['conversation_id']}",
                            headers=_auth_headers(), timeout=5,
                        )
                        if r2.status_code == 200:
                            msgs = r2.json()
                            st.session_state.messages = []
                            st.session_state.conversation_id = c["conversation_id"]
                            for m in msgs:
                                if m["role"] == "user":
                                    st.session_state.messages.append({"role": "user", "content": m["content"]})
                                else:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "answer": m["content"],
                                        "sources": json.dumps(m.get("sources") or []),
                                    })
                            st.rerun()
        except Exception:
            st.caption("Could not load history")

        st.divider()

        # Favorites
        st.subheader("Favorites")
        try:
            resp = httpx.get(f"{BACKEND_URL}/favorites", headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                favs = resp.json()
                for f in favs[:5]:
                    with st.expander(textwrap.shorten(f["question"], 40)):
                        st.markdown(f["answer"][:200] + "...")
        except Exception:
            st.caption("Could not load favorites")

        st.divider()

        # Preferences
        st.subheader("Preferences")
        try:
            resp = httpx.get(f"{BACKEND_URL}/preferences", headers=_auth_headers(), timeout=5)
            if resp.status_code == 200:
                prefs = resp.json()
                new_theme = st.selectbox("Theme", ["light", "dark"], index=0 if prefs["theme"] == "light" else 1)
                new_topk = st.slider("Top K results", 1, 10, prefs["top_k"])
                if st.button("Save Preferences"):
                    httpx.put(
                        f"{BACKEND_URL}/preferences",
                        headers=_auth_headers(),
                        json={"theme": new_theme, "top_k": new_topk},
                        timeout=5,
                    )
                    st.success("Saved!")
        except Exception:
            st.caption("Could not load preferences")


# ═══════════════════════════════════════════════════════════════════════════
# Main app (authenticated)
# ═══════════════════════════════════════════════════════════════════════════

def show_main_app():
    show_sidebar()
    st.title("SoloRAG -- Stripe FAQ Assistant")

    tab_rag, tab_pokemon = st.tabs(["RAG Chat", "Pokemon"])

    # ── RAG Chat ──────────────────────────────────────────────────────
    with tab_rag:
        for idx, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["answer"])
                    col_src, col_fav = st.columns([4, 1])
                    with col_src:
                        with st.expander("Show Sources"):
                            st.markdown(msg.get("sources", ""))
                    with col_fav:
                        if idx in st.session_state.favorited:
                            st.markdown("**Favorited**")
                        else:
                            # Find the preceding user question for this answer
                            question_text = ""
                            if idx > 0 and st.session_state.messages[idx - 1]["role"] == "user":
                                question_text = st.session_state.messages[idx - 1]["content"]
                            if st.button("Favorite", key=f"fav_{idx}"):
                                try:
                                    resp = httpx.post(
                                        f"{BACKEND_URL}/favorites",
                                        headers=_auth_headers(),
                                        json={
                                            "question": question_text,
                                            "answer": msg["answer"],
                                            "sources": [],
                                        },
                                        timeout=5,
                                    )
                                    if resp.status_code == 201:
                                        st.session_state.favorited.add(idx)
                                        st.toast("Saved to favorites!")
                                        st.rerun()
                                    else:
                                        st.error("Could not save favorite")
                                except Exception:
                                    st.error("Could not save favorite")

        if prompt := st.chat_input("Ask a question about Stripe payments..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer_box = st.empty()
                sources_container = st.container()

            async def fetch_stream(question: str):
                retries = 8
                backoff = 0.5
                last_exc: Union[None, Exception] = None
                for attempt in range(retries):
                    try:
                        async with httpx.AsyncClient(timeout=None) as client:
                            async with client.stream(
                                "POST",
                                f"{BACKEND_URL}/query/stream",
                                json={"question": question, "conversation_id": st.session_state.conversation_id},
                                headers=_auth_headers(),
                            ) as r:
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
                                                f"* {textwrap.shorten(s['text'], 120)} (score: {s['score']:.2f})"
                                                for s in src_json
                                            )
                                        except Exception:
                                            pass
                                        with sources_container.expander("Show Sources"):
                                            st.markdown(formatted_raw)
                                        return body, formatted_raw
                                    answer_box.markdown(buffer + " ▌")
                    except Exception as e:
                        last_exc = e
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 5)
                        continue
                if last_exc:
                    st.error(f"Backend not ready yet: {last_exc}")
                return "", ""

            body, formatted = asyncio.run(fetch_stream(prompt))
            st.session_state.messages.append({"role": "assistant", "answer": body, "sources": formatted})
            st.rerun()  # rerun so the history loop renders the latest message with the favorite button

    # ── Pokemon Tab ───────────────────────────────────────────────────
    with tab_pokemon:
        st.subheader("Pokemon Lookup")
        st.caption("Powered by PokeAPI -- free, no API key needed")

        pokemon_name = st.text_input("Enter a Pokemon name or Pokedex number", placeholder="e.g. pikachu, charizard, 25")
        if st.button("Look Up", type="primary"):
            if not pokemon_name.strip():
                st.warning("Please enter a Pokemon name or number.")
            else:
                with st.spinner("Fetching Pokemon data..."):
                    try:
                        resp = httpx.get(f"{BACKEND_URL}/pokemon", params={"name": pokemon_name.strip()}, timeout=10)
                        data = resp.json()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        data = {"error": str(e)}

                if "error" in data:
                    st.error(data["error"])
                else:
                    col_img, col_info = st.columns([1, 3])
                    with col_img:
                        if data.get("sprite"):
                            st.image(data["sprite"], width=120)
                    with col_info:
                        st.markdown(f"### #{data['id']} -- {data['name']}")
                        st.markdown("**Types:** " + ", ".join(data.get("types", [])))
                        st.markdown("**Abilities:** " + ", ".join(data.get("abilities", [])))

                    col_h, col_w = st.columns(2)
                    col_h.metric("Height", f"{data['height_m']} m")
                    col_w.metric("Weight", f"{data['weight_kg']} kg")

                    st.markdown("#### Base Stats")
                    stats = data.get("stats", {})
                    stat_labels = {
                        "hp": "HP", "attack": "Attack", "defense": "Defense",
                        "special-attack": "Sp. Atk", "special-defense": "Sp. Def", "speed": "Speed",
                    }
                    cols = st.columns(len(stat_labels))
                    for col, (key, label) in zip(cols, stat_labels.items()):
                        col.metric(label, stats.get(key, "--"))


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.token:
    show_main_app()
else:
    show_auth_page()
