"""
app.py

Phase 3: Streamlit Frontend for Local RAG Research Assistant.
Provides a chat UI connected to retriever.py and rag_pipeline.py, complete with
domain filtering, session history support, and paper citation expanders.
"""

import streamlit as st
from modules.generation.rag_pipeline import generate_answer

# Page Configuration
st.set_page_config(
    page_title="Local RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Sidebar Configuration
with st.sidebar:
    st.title("📚 Research Assistant")
    st.markdown(
        "A local Retrieval-Augmented Generation (RAG) tool querying 300+ arXiv papers "
        "across Cybersecurity and AI/ML."
    )
    st.markdown("---")
    
    # Domain Filtering
    st.subheader("Filter by Domain")
    domain_choice = st.radio(
        "Select topic scope:",
        options=["Both", "Cybersecurity", "AI/ML"],
        index=0,
        help="Restrict vector search to papers within a specific domain or search both."
    )
    
    # Map domain selection to backend domain filter string
    domain_map = {
        "Both": "both",
        "Cybersecurity": "cybersecurity",
        "AI/ML": "ai_ml"
    }
    selected_domain_filter = domain_map[domain_choice]
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Powered by ChromaDB, BAAI/bge-base-en-v1.5 & Groq LLaMA-3.3-70B")

# Main Header
st.title("🔬 Scientific Paper Research Assistant")
st.caption(f"Active Search Scope: **{domain_choice}**")

# Initialize Session State Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render Sources Expander for assistant messages if present
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources & Citations", expanded=False):
                for idx, src in enumerate(message["sources"], 1):
                    title = src.get("title", "Untitled Paper")
                    page = src.get("page", "?")
                    source_url = src.get("source_url", "")
                    domain = src.get("domain", "").upper()
                    subtopic = src.get("subtopic", "")
                    
                    title_md = f"[{title}]({source_url})" if source_url else f"**{title}**"
                    
                    st.markdown(
                        f"**{idx}. {title_md}**  \n"
                        f"📄 Page {page} | Domain: `{domain}` | Subtopic: `{subtopic}`"
                    )
                    if source_url:
                        st.markdown(f"🔗 **arXiv Link**: [{source_url}]({source_url})")
                        
                    if idx < len(message["sources"]):
                        st.markdown("---")

# Handle User Input
if prompt := st.chat_input("Ask a question about cybersecurity or AI/ML research papers..."):
    # Append user turn to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant turn
    with st.chat_message("assistant"):
        with st.spinner("Searching research papers & generating answer..."):
            # Format chat history context for multi-turn follow-ups
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            
            result = generate_answer(
                user_question=prompt,
                chat_history=history_payload,
                domain_filter=selected_domain_filter
            )
            
            answer_text = result["answer"]
            sources = result["sources"]
            
            st.markdown(answer_text)
            
            if sources:
                with st.expander("📚 Sources & Citations", expanded=False):
                    for idx, src in enumerate(sources, 1):
                        title = src.get("title", "Untitled Paper")
                        page = src.get("page", "?")
                        source_url = src.get("source_url", "")
                        domain = src.get("domain", "").upper()
                        subtopic = src.get("subtopic", "")
                        
                        title_md = f"[{title}]({source_url})" if source_url else f"**{title}**"
                        
                        st.markdown(
                            f"**{idx}. {title_md}**  \n"
                            f"📄 Page {page} | Domain: `{domain}` | Subtopic: `{subtopic}`"
                        )
                        if source_url:
                            st.markdown(f"🔗 **arXiv Link**: [{source_url}]({source_url})")
                            
                        if idx < len(sources):
                            st.markdown("---")
                            
            # Append assistant turn with sources to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })
