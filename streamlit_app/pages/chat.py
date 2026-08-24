"""
Chat page.

Reachable only with a valid access token in session state; otherwise the user
is sent back to the sign-in page.
"""

import streamlit as st

st.set_page_config(
    page_title="Adaptive RAG - Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.api_client import (  # noqa: E402
    ApiError,
    query_backend,
    upload_document,
)

HOME_PAGE = "home.py"

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Authentication guard --------------------------------------------------
token = st.session_state.get("access_token")
if not token:
    st.warning("Please sign in first.")
    if st.button("Go to sign in"):
        st.switch_page(HOME_PAGE)
    st.stop()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploaded_files", {})
st.session_state.setdefault("show_logout_confirm", False)

# --- Header ----------------------------------------------------------------
header, actions = st.columns([8, 2])
with header:
    st.title("💬 Adaptive RAG Chat")
    st.caption(f"Signed in as {st.session_state.get('username', 'user')}")
with actions:
    if st.button("🔒 Logout", use_container_width=True):
        st.session_state.show_logout_confirm = True

if st.session_state.show_logout_confirm:
    st.warning("Sign out of this session?")
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button("✅ Yes, sign out", use_container_width=True):
            st.session_state.clear()
            st.switch_page(HOME_PAGE)
    with cancel:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_logout_confirm = False
            st.rerun()

# --- Document upload -------------------------------------------------------
with st.sidebar:
    st.header("📂 Your documents")
    st.caption("Documents are private to your account.")

    uploaded_file = st.file_uploader("PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        description = st.text_input(
            "📄 Describe this document (required)",
            max_chars=300,
            placeholder="E.g. my CV, or a LangGraph tutorial",
        )

        if not description:
            st.info("Add a short description to enable upload.")
        else:
            file_key = f"{uploaded_file.name}:{uploaded_file.size}:{description}"
            if file_key in st.session_state.uploaded_files:
                st.success(f"Already indexed: {uploaded_file.name}")
            elif st.button("Upload and index", type="primary", use_container_width=True):
                with st.spinner("Indexing document..."):
                    try:
                        result = upload_document(uploaded_file, description, token)
                    except ApiError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.uploaded_files[file_key] = result
                        st.success(
                            f"Indexed {result['filename']} "
                            f"({result['chunks_indexed']} chunks)."
                        )

    if st.session_state.uploaded_files:
        st.divider()
        st.caption("Indexed this session:")
        for result in st.session_state.uploaded_files.values():
            st.write(f"• {result['filename']} ({result['chunks_indexed']} chunks)")

# --- Conversation ----------------------------------------------------------
for role, text in st.session_state.chat_history:
    st.chat_message(role).write(text)

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = query_backend(
                    user_input, st.session_state["session_id"], token
                )
            except ApiError as exc:
                answer = f"⚠️ {exc}"
        st.write(answer)

    st.session_state.chat_history.append(("assistant", answer))
