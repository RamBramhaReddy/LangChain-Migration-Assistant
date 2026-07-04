import json
import time

import streamlit as st

from pipeline import run_pipeline


st.set_page_config(
    page_title="LangChain Migration Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Inter:wght@400;500;600;700&display=swap');

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0a0e16;
}

.main .block-container {
    max-width: 1150px;
    padding-top: 3rem;
    padding-bottom: 4rem;
    margin: 0 auto;
}

/* ---------- HERO ---------- */

.hero-center {
    text-align: center;
    margin-bottom: 2.2rem;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(45, 212, 191, 0.08);
    color: #2dd4bf;
    border: 1px solid rgba(45, 212, 191, 0.3);
    border-radius: 4px;
    padding: 0.3rem 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
}

.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.6rem;
    font-weight: 800;
    color: #f5f8fb;
    margin: 0 0 0.7rem 0;
    line-height: 1.2;
    letter-spacing: -0.01em;
}

.hero-subtitle {
    font-size: 1rem;
    color: #8592a3;
    max-width: 600px;
    line-height: 1.6;
    margin: 0 auto 1.6rem auto;
}

/* ---------- CHIPS ---------- */

.chip-row {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 2rem;
}

.chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #a9b4c0;
    background: #11161f;
    border: 1px solid #232b38;
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
}

/* ---------- INPUT ---------- */

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #556173;
    margin: 0 0 0.6rem 0;
}

.section-label::before {
    content: "> ";
    color: #2dd4bf;
}

.stTextArea textarea {
    background-color: #0d121b !important;
    border: 1px solid #232b38 !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
}

.stTextArea textarea:focus {
    border-color: #2dd4bf !important;
    box-shadow: 0 0 0 1px #2dd4bf !important;
}

/* ---------- BUTTONS ---------- */

.stButton button {
    border-radius: 8px;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 0;
    transition: all 0.15s ease;
}

button[kind="primary"] {
    background: #2dd4bf !important;
    color: #0a0e16 !important;
    border: none !important;
}

button[kind="primary"]:hover {
    background: #5eead4 !important;
}

button[kind="secondary"] {
    background: transparent !important;
    color: #8592a3 !important;
    border: 1px solid #232b38 !important;
}

button[kind="secondary"]:hover {
    border-color: #556173 !important;
    color: #c3ccd6 !important;
}

button:focus-visible {
    outline: 2px solid #2dd4bf !important;
    outline-offset: 2px !important;
}

/* ---------- STAT CARDS ---------- */

.stat-card {
    background: #11161f;
    border: 1px solid #232b38;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}

.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #556173;
    margin-bottom: 0.3rem;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #2dd4bf;
}

/* ---------- RESULT CARDS ---------- */

.result-card {
    background: #11161f;
    border: 1px solid #232b38;
    border-radius: 12px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
}

.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f5f8fb;
    margin-bottom: 0.7rem;
}

.code-tab {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8592a3;
    background: #161c27;
    border: 1px solid #232b38;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 0.35rem 0.8rem;
    margin-top: 0.3rem;
}

.code-tab::before {
    content: "●";
    color: #2dd4bf;
    font-size: 0.6rem;
}

/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1.6rem;
    border-top: 1px solid #1b222e;
}

.footer-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #556173;
    margin-bottom: 0.5rem;
}

.footer-stack {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8592a3;
}
</style>
"""

st.markdown(custom_style, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-center">
        <span class="eyebrow">RAG &middot; VERSION-AWARE MIGRATION</span>
        <div class="hero-title">LangChain Migration Assistant</div>
        <div class="hero-subtitle">
            Paste a code snippet, stack trace, or migration question and get a
            version-aware answer grounded in official LangChain documentation.
        </div>
        <div class="chip-row">
            <span class="chip">LangChain Code</span>
            <span class="chip">Import Errors</span>
            <span class="chip">Stack Traces</span>
            <span class="chip">Migration Questions</span>
            <span class="chip">Version Questions</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="section-label">Paste your code, error, or question</div>', unsafe_allow_html=True)

query = st.text_area(
    label="query_input",
    label_visibility="collapsed",
    height=220,
    placeholder="""from langchain.chat_models import ChatOpenAI

ImportError: cannot import name 'ChatOpenAI'

What replaced BaseLanguageModel.predict()?
"""
)

_, col_a, col_b, _ = st.columns([3, 1, 1, 3])

with col_a:
    analyze = st.button("Analyze Migration", type="primary", use_container_width=True)

with col_b:
    clear = st.button("Clear", type="secondary", use_container_width=True)

if clear:
    st.rerun()


def render_card(title, content, color, is_code=False, code_label="migrated.py"):

    st.markdown(
        f'<div class="result-card" style="border-left:4px solid {color}">'
        f'<div class="card-title">{title}</div>',
        unsafe_allow_html=True
    )

    if is_code:
        st.markdown(f'<div class="code-tab">{code_label}</div>', unsafe_allow_html=True)
        st.code(content, language="python")
    else:
        st.markdown(content)

    st.markdown('</div>', unsafe_allow_html=True)


def render_deprecated_apis_card(deprecated_apis):

    st.markdown(
        '<div class="result-card" style="border-left:4px solid #f0a94e">'
        '<div class="card-title">Deprecated APIs</div>',
        unsafe_allow_html=True
    )

    if deprecated_apis:
        st.table(
            [
                {
                    "Old API": item.get("old_api", ""),
                    "Replacement": item.get("replacement", ""),
                    "Reason": item.get("reason", ""),
                }
                for item in deprecated_apis
            ]
        )
    else:
        st.markdown("No deprecated APIs detected.")

    st.markdown('</div>', unsafe_allow_html=True)


if analyze:

    if not query.strip():

        st.warning("Please enter a question, code snippet, or error message.")

    else:

        with st.status("Analyzing migration...", expanded=True) as status:

            st.write("✓ Detecting input type")
            time.sleep(0.25)

            st.write("✓ Retrieving LangChain docs")
            time.sleep(0.25)

            st.write("✓ Hybrid retrieval (dense + BM25)")
            time.sleep(0.25)

            st.write("✓ Cross-encoder reranking")
            time.sleep(0.25)

            st.write("✓ Generating migration guide")

            try:
                result = run_pipeline(query)

                answer = result["answer"]
                contexts = result["contexts"]
                documents = result["documents"]

                status.update(label="Migration analysis complete", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Analysis failed", state="error", expanded=True)
                st.error(str(e))
                answer = None
                contexts = []
                documents = []

        if answer:

            # pipeline.py returns structured JSON. It may already be a dict,
            # or (in its raw-text fallback case) a plain string. Handle both
            # without any regex or heading parsing.
            if isinstance(answer, dict):
                data = answer
            else:
                try:
                    data = json.loads(answer)
                except (json.JSONDecodeError, TypeError):
                    data = None

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    '<div class="stat-card"><div class="stat-label">Status</div>'
                    '<div class="stat-value">Success</div></div>',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(
                    '<div class="stat-card"><div class="stat-label">Source</div>'
                    f'<div class="stat-value">{len(documents)} Docs</div></div>',
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    '<div class="stat-card"><div class="stat-label">Retrieval</div>'
                    '<div class="stat-value">Hybrid + Rerank</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if data:

                if data.get("summary"):
                    render_card("Summary", data["summary"], "#8592a3")

                if "deprecated_apis" in data:
                    render_deprecated_apis_card(data.get("deprecated_apis") or [])

                if data.get("updated_code"):
                    render_card(
                        "Updated Code",
                        data["updated_code"],
                        "#2dd4bf",
                        is_code=True,
                        code_label="migrated.py"
                    )

                if data.get("notes"):
                    render_card("Notes", data["notes"], "#8592a3")

                if data.get("example"):
                    render_card(
                        "Example",
                        data["example"],
                        "#8592a3",
                        is_code=True,
                        code_label="example.py"
                    )

            else:
                # pipeline.py's documented fallback: JSON parsing failed
                # upstream and raw text was returned instead. No regex,
                # no heading splitting — just show it plainly.
                render_card("Result", answer, "#2dd4bf")

            if documents:
                with st.expander("Retrieved Sources"):
                    for i, doc in enumerate(documents, start=1):
                        st.markdown(f"### Source {i}")

                        metadata = getattr(doc, "metadata", None)
                        if metadata is None and isinstance(doc, dict):
                            metadata = doc.get("metadata", {})
                        metadata = metadata or {}

                        if metadata:
                            st.markdown(
                                f"**Version:** {metadata.get('version', 'Unknown')}  \n"
                                f"**Document:** {metadata.get('doc_type', 'Unknown')}  \n"
                                f"**Section:** {metadata.get('section', 'Unknown')}  \n"
                                f"**Source:** {metadata.get('source', 'Unknown')}"
                            )
                            st.json(metadata)
                        else:
                            st.json({})

                        content = getattr(doc, "page_content", None)
                        if content is None and isinstance(doc, dict):
                            content = doc.get("page_content", "")
                        content = content or ""

                        preview = content[:600]
                        if len(content) > 600:
                            preview += "..."
                        st.markdown(preview)

st.markdown(
    """
    <div class="footer">
        <div class="footer-label">Powered by</div>
        <div class="footer-stack">LangChain &middot; Qdrant &middot; BM25 &middot; Cross-Encoder &middot; GPT-OSS 120B</div>
    </div>
    """,
    unsafe_allow_html=True
)