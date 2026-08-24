"""
Login and registration page.

``st.set_page_config`` must be the first Streamlit call on the page; anything
before it raises ``StreamlitSetPageConfigMustBeFirstCommandError``.
"""

import streamlit as st

st.set_page_config(page_title="Adaptive RAG - Sign in", page_icon="🔐")

import uuid  # noqa: E402

from utils.api_client import ApiError, api_available, login, register  # noqa: E402

CHAT_PAGE = "pages/chat.py"

# Hide the automatic page navigation so the chat page is reachable only after
# signing in.
st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔐 Adaptive RAG")
st.caption("Sign in to chat with your own documents.")

if st.session_state.get("access_token"):
    st.success(f"Signed in as {st.session_state.get('username', 'user')}.")
    if st.button("Go to chat", type="primary"):
        st.switch_page(CHAT_PAGE)
    st.stop()

if not api_available():
    st.error(
        "The API is not reachable. Start it with:\n\n"
        "`uvicorn src.main:app --host 127.0.0.1 --port 8000`"
    )
    st.stop()

mode = st.radio("Choose action:", ["Login", "Create account"], horizontal=True)

with st.form("auth_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    if not username or not password:
        st.error("Username and password are required.")
    else:
        try:
            result = (
                register(username, password)
                if mode == "Create account"
                else login(username, password)
            )
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.session_state["access_token"] = result["access_token"]
            st.session_state["username"] = result["username"]
            # A conversation id, distinct from the auth token.
            st.session_state["session_id"] = uuid.uuid4().hex
            st.session_state["chat_history"] = []
            st.switch_page(CHAT_PAGE)

with st.expander("Requirements"):
    st.markdown(
        "- Username: 3-64 characters, letters, digits, `.`, `_` or `-`\n"
        "- Password: at least 8 characters"
    )
