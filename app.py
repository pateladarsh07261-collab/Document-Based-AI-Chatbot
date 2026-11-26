import glob
import os
import time
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import memory_manager as mm  # [DAY 4] SQLite Short-Term, Long-Term & User Profile Manager

st.set_page_config(page_title="Humaeli — Personalized Mental Wellness AI", page_icon="🧠", layout="wide")
st.title("🧠 Humaeli — Personalized Mental Wellness AI Assistant")

KB_DIR = "knowledge_base"
DB_DIR = "chroma_db"

# Initialize SQLite Database & 3 Dummy Profiles (user_101, user_102, user_103)
mm.init_db()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_llm(model_name, temperature, top_p, max_tokens):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=1.25,  # Repetition loop rokne ke liye
        num_predict=max_tokens,
    )


embedding = load_embeddings()
reranker = load_reranker()


# --- 1. Knowledge Base Loader (Day 3) ---
def load_kb_documents():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for path in glob.glob(os.path.join(KB_DIR, "*.txt")):
        fname = os.path.basename(path)
        category = fname.replace(".txt", "").replace("_", " ").title()
        loader = TextLoader(path, encoding="utf-8")
        for doc in loader.load():
            doc.page_content = " ".join(doc.page_content.split())
            doc.metadata = {"source": fname, "category": category, "page": 0}
            chunks.extend(splitter.split_documents([doc]))
    return chunks


def get_vectorstore():
    vs = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
    if vs._collection.count() == 0:
        kb_docs = load_kb_documents()
        if kb_docs:
            vs.add_documents(kb_docs)
    return vs


vectorstore = get_vectorstore()

# --- 2. Sidebar: User Profile, Create New User, Privacy Controls & Sessions ---
users_list = mm.get_all_users()
user_options = {f"{u['user_id']} — {u['name']}": u["user_id"] for u in users_list}

with st.sidebar:
    st.header("👤 User Profile & Privacy")
    selected_label = st.selectbox("Select User Account", list(user_options.keys()), key="select_user_box")
    current_user_id = user_options[selected_label]
    profile = mm.get_profile(current_user_id)

    # Create New User Account Form (Unique keys used to avoid DuplicateElementId)
    with st.expander("➕ Create New User", expanded=False):
        new_uid = st.text_input("New User ID", placeholder="e.g. user_104", key="new_uid")
        new_name = st.text_input("Full Name", placeholder="e.g. Adarsh Patidar", key="new_name")
        new_age = st.text_input("Age Group", value="18-24 (College Student)", key="new_age")
        new_lang = st.selectbox("Language", ["Hinglish", "English", "Hindi"], key="new_lang")
        new_interests = st.text_input("Interests", placeholder="e.g. Coding, Cricket, Music", key="new_interests")
        new_style = st.text_input("Communication Style", value="Warm, concise bullet points", key="new_style")
        new_concerns = st.text_input("Initial Concerns", placeholder="e.g. Project deadline stress", key="new_concerns")
        new_mem = st.toggle("Enable Memory", value=True, key="new_mem")

        if st.button("Create User Account", use_container_width=True, key="btn_create_user"):
            if not new_uid.strip() or not new_name.strip():
                st.error("⚠️ User ID aur Full Name dono zaroori hain.")
            elif mm.create_user(new_uid, new_name, new_age, new_lang, new_interests, new_style, new_concerns, new_mem):
                st.success(f"✅ User '{new_name}' ({new_uid}) ban gaya!")
                st.rerun()
            else:
                st.error("❌ Yeh User ID pehle se exist karta hai, koi doosra ID daalein.")

    # User switch hone par session reset karo taaki privacy isolation bana rahe
    if st.session_state.get("active_user") != current_user_id:
        st.session_state.active_user = current_user_id
        user_sessions = mm.get_user_sessions(current_user_id)
        st.session_state.current_session = user_sessions[0]

    # Edit Existing User Profile & Memory Toggle (Unique keys used)
    with st.expander("📝 Edit Profile & Preferences", expanded=False):
        age_grp = st.text_input("Age Group", value=profile["age_group"], key="edit_age")
        lang_choices = ["Hinglish", "English", "Hindi"]
        lang_idx = lang_choices.index(profile["preferred_language"]) if profile["preferred_language"] in lang_choices else 0
        pref_lang = st.selectbox("Preferred Language", lang_choices, index=lang_idx, key="edit_lang")
        interests = st.text_input("Interests", value=profile["interests"], key="edit_interests")
        comm_style = st.text_input("Communication Style", value=profile["communication_style"], key="edit_style")
        past_concerns = st.text_input("Previously Discussed Concerns", value=profile["past_concerns"], key="edit_concerns")
        mem_toggle = st.toggle("Enable Long-Term Memory", value=bool(profile["memory_enabled"]), key="edit_mem")

        if st.button("Save Profile Settings", use_container_width=True, key="btn_save_profile"):
            mm.update_profile(
                current_user_id, age_grp, pref_lang, interests, comm_style, past_concerns, mem_toggle
            )
            st.success("✅ Profile & memory preference saved!")
            st.rerun()

    st.caption(
        f"**Lang:** `{profile['preferred_language']}` | **Memory:** `{'ON' if profile['memory_enabled'] else 'OFF'}`"
    )
    st.info(f"**Conversation Summary:** {profile['conversation_summary']}")

    # View & Delete Individual Long-Term Memories
    with st.expander("🧠 Saved Long-Term Memories", expanded=True):
        saved_mems = mm.get_all_memories(current_user_id)
        if not saved_mems:
            st.caption("No long-term memories saved for this user yet.")
        else:
            for m in saved_mems:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"- {m['fact']} *({m['created_at']})*")
                if col2.button("🗑️", key=f"del_{current_user_id}_{m['id']}", help="Delete memory"):
                    mm.delete_memory_by_id(current_user_id, m["id"])
                    st.rerun()

    st.divider()
    st.header("💬 User Sessions")
    user_sessions = mm.get_user_sessions(current_user_id)
    if st.session_state.current_session not in user_sessions:
        user_sessions.append(st.session_state.current_session)

    selected_session = st.selectbox(
        "Active Session",
        user_sessions,
        index=user_sessions.index(st.session_state.current_session),
        key="select_session_box",
    )
    st.session_state.current_session = selected_session

    new_session_name = st.text_input("New Session Name", placeholder="e.g. session_2", key="new_session_input")
    if st.button("Create Session", use_container_width=True, key="btn_create_session") and new_session_name.strip():
        st.session_state.current_session = new_session_name.strip()
        st.rerun()

    if st.button("Clear Current Session Chat", use_container_width=True, key="btn_clear_chat"):
        mm.clear_session_history(current_user_id, st.session_state.current_session)
        st.rerun()

    st.divider()
    st.header("⚙️ RAG & LLM Settings")
    selected_model = st.selectbox("Model", ["qwen2.5:3b", "llama3.2:latest", "qwen2.5:1.5b"], key="model_box")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.25, step=0.05, key="temp_slider")
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9, step=0.1, key="top_p_slider")
    max_tokens = st.slider("Max Tokens", 128, 512, 220, step=32, key="max_tokens_slider")

    raw_meta = vectorstore.get().get("metadatas", [])
    available_cats = sorted({m.get("category") for m in raw_meta if m and m.get("category")})
    selected_category = st.selectbox("Category Filter", ["All Categories"] + available_cats, key="cat_box")
    top_k = st.slider("Top-K Chunks", 2, 6, 3, key="topk_slider")

    st.divider()
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF ya TXT", type=["pdf", "txt"], accept_multiple_files=True, key="file_up")
    if st.button("Process & Add", use_container_width=True, key="btn_upload") and uploaded_files:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        new_chunks = []
        for file in uploaded_files:
            if file.size == 0:
                continue
            suffix = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            os.remove(tmp_path)
            cat_name = os.path.splitext(file.name)[0].replace("_", " ").title()
            for doc in docs:
                cleaned = " ".join(doc.page_content.split())
                if cleaned:
                    doc.page_content = cleaned
                    doc.metadata["source"] = file.name
                    doc.metadata["category"] = cat_name
                    new_chunks.extend(splitter.split_documents([doc]))
        if new_chunks:
            vectorstore.add_documents(new_chunks)
            st.success(f"✅ {len(new_chunks)} chunks added!")


llm = get_llm(selected_model, temperature, top_p, max_tokens)

# --- 3. Prompts ---
rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the Question into a clear standalone search query using Chat History so pronouns like 'it' or 'that' are resolved. "
    "Return ONLY the standalone query on one line.\n\nChat History:\n{chat_history}\n\nQuestion: {question}\nStandalone Query:"
)

summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the user's key concerns, shared facts, and emotional context in 2-3 concise English sentences. "
    "Do NOT invent details.\n\nPrevious Summary: {old_summary}\nRecent Chat:\n{recent_chat}\n\nUpdated Summary:"
)

qa_prompt = ChatPromptTemplate.from_template(
    "You are Humaeli, a warm and supportive AI Mental Wellness Educational Assistant.\n"
    "USER PROFILE:\n"
    "- Name: {user_name}\n"
    "- Age Group: {age_group}\n"
    "- Preferred Language: {preferred_language}\n"
    "- Communication Style: {communication_style}\n"
    "- Interests: {interests}\n"
    "- Past Concerns: {past_concerns}\n"
    "- Conversation Summary: {conversation_summary}\n\n"
    "RETRIEVED LONG-TERM MEMORIES (Cross-Session):\n{lt_memories}\n\n"
    "SHORT-TERM SESSION HISTORY:\n{chat_history}\n\n"
    "KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
    "LANGUAGE INSTRUCTION (STRICT):\n"
    "- If Preferred Language is 'Hinglish', write strictly using ONLY ASCII English letters (A-Z, a-z, 0-9) in friendly conversational style like: 'Rohan, exam ka stress normal hai. Aap ye 3 simple tips follow karein:'. NEVER use Hindi Devanagari letters.\n"
    "- If Preferred Language is 'English', reply strictly in clear English.\n\n"
    "RULES:\n"
    "1. For greetings, introductions, or questions about user profile/memories, reply warmly using ONLY exact facts from User Profile, Memories, and Chat History—NEVER guess an exact age or number not explicitly written in Profile. Never say 'Not found in the knowledge base.'\n"
    "2. If the user asks about past feelings or why they were stressed, directly state the exact reason from RETRIEVED LONG-TERM MEMORIES first, then give 3 short bullet points from Knowledge Base Context.\n"
    "3. For mental wellness questions, answer strictly using Knowledge Base Context in 3-4 concise bullet points (under 80 words total). Only if an unrelated factual/medical topic is completely missing from Context, reply: 'Not found in the knowledge base.'\n"
    "4. SAFETY RULE: Provide educational wellness tips only. Never diagnose or prescribe medication.\n\n"
    "Question: {question}\n"
    "Answer:"
)


# --- 4. Helper Functions for Auto-Summarization & Fact Extraction ---
def maybe_summarize(user_id, session_id, profile):
    msgs = mm.get_session_messages(user_id, session_id)
    if len(msgs) >= 8 and len(msgs) % 4 == 0:
        recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[-8:])
        new_sum = (summary_prompt | llm | StrOutputParser()).invoke(
            {"old_summary": profile["conversation_summary"], "recent_chat": recent_text}
        ).strip()
        mm.update_summary(user_id, new_sum)
        st.toast("📝 Conversation Summary Auto-Updated!")


def extract_and_store_fact(user_id, question, profile):
    if not profile["memory_enabled"]:
        return
    personal_markers = (
        "my name", "i am", "i'm", "i feel", "i have", "my exam", "my work",
        "mera naam", "mujhe", "main ", "i like", "i prefer", "stress about", "trouble sleeping"
    )
    if any(m in question.lower() for m in personal_markers) and len(question.split()) >= 3:
        mm.save_long_term_memory(user_id, f"User shared: {question.strip()}", embedding)


# --- 5. Main Chat Stream Pipeline ---
def ask_stream(question, user_id, session_id):
    profile = mm.get_profile(user_id)
    session_msgs = mm.get_session_messages(user_id, session_id)
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in session_msgs[-6:]) if session_msgs else "None"

    start_time = time.time()

    # Cross-Session Long-Term Memory Retrieval (Cosine Similarity >= 0.28)
    lt_memories = mm.search_long_term_memories(user_id, question, embedding, min_score=0.28, top_k=3)
    lt_mem_text = "\n".join(f"- {m}" for m in lt_memories) if lt_memories else "None"

    # Follow-Up Query Rewriting
    follow_up_words = ("it", "its", "that", "this", "they", "them", "those", "more", "explain", "kaise", "kya")
    is_followup = any(w in question.lower().split() for w in follow_up_words)
    if is_followup and (session_msgs or lt_memories):
        combined_hist = f"Memories:\n{lt_mem_text}\nRecent Chat:\n{hist_text}"
        search_query = (rewrite_prompt | llm | StrOutputParser()).invoke(
            {"chat_history": combined_hist, "question": question}
        ).strip()
        st.caption(f"🔎 **Rewritten Query:** `{search_query}`")
    else:
        search_query = question

    # Step 1: Hybrid Search (BM25 + Semantic RRF) + Metadata Filter
    raw = vectorstore.get()
    all_docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(raw["documents"], raw["metadatas"])
        if selected_category == "All Categories" or m.get("category") == selected_category
    ]

    bm25_ret = BM25Retriever.from_documents(all_docs)
    bm25_ret.k = 8
    search_kwargs = {"k": 8}
    if selected_category != "All Categories":
        search_kwargs["filter"] = {"category": selected_category}
    vec_ret = vectorstore.as_retriever(search_kwargs=search_kwargs)

    bm25_docs = bm25_ret.invoke(search_query)
    vec_docs = vec_ret.invoke(search_query)

    scores_map, doc_map = {}, {}
    for rank, d in enumerate(bm25_docs):
        doc_map[d.page_content] = d
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.4 / (rank + 60)
    for rank, d in enumerate(vec_docs):
        doc_map[d.page_content] = d
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.6 / (rank + 60)
    candidate_docs = [doc_map[k] for k, _ in sorted(scores_map.items(), key=lambda x: x[1], reverse=True)][:8]

    # Step 2: Cross-Encoder Reranking (Top-8 -> Top-3)
    if candidate_docs:
        pairs = [[search_query, d.page_content] for d in candidate_docs]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
        relevant_docs = [doc for _, doc in ranked[:top_k]]
    else:
        relevant_docs = []

    personal_triggers = ("hi", "hello", "hey", "my name", "my age", "who am i", "what is my")
    if any(question.lower().startswith(p) for p in personal_triggers):
        context = "Personal greeting/introduction. Reply warmly using User Profile, Memories, and Chat History."
    else:
        context = (
            "\n\n".join(f"[{d.metadata.get('category')}] {d.page_content}" for d in relevant_docs)
            if relevant_docs
            else "No matching notes found."
        )

    chain = qa_prompt | llm | StrOutputParser()
    answer = st.write_stream(
        chain.stream({
            "user_name": profile["name"],
            "age_group": profile["age_group"],
            "preferred_language": profile["preferred_language"],
            "communication_style": profile["communication_style"],
            "interests": profile["interests"],
            "past_concerns": profile["past_concerns"],
            "conversation_summary": profile["conversation_summary"],
            "lt_memories": lt_mem_text,
            "chat_history": hist_text,
            "context": context,
            "question": question,
        })
    ).strip()

    elapsed = round(time.time() - start_time, 2)
    sources_list = sorted({f"`{d.metadata.get('source', 'Unknown')}`" for d in relevant_docs})
    source_str = ", ".join(sources_list) if sources_list else "`Memory/Profile`"
    footer = (
        f"\n\n👤 `{profile['name']}` (`{profile['preferred_language']}`) | "
        f"🧠 Memories Used: `{len(lt_memories)}` | 📚 {source_str} | ⚡ `{elapsed}s`"
    )
    st.markdown(footer)

    # Save Short-Term Messages, Extract Long-Term Fact & Trigger Auto-Summarization
    mm.add_message(user_id, session_id, "user", question)
    mm.add_message(user_id, session_id, "assistant", answer + footer)
    extract_and_store_fact(user_id, question, profile)
    maybe_summarize(user_id, session_id, profile)


# Render Active User + Session Messages
active_msgs = mm.get_session_messages(current_user_id, st.session_state.current_session)
for m in active_msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_prompt := st.chat_input("Chat with Humaeli (personalized to your profile & memory)..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.chat_message("assistant"):
        ask_stream(user_prompt, current_user_id, st.session_state.current_session)