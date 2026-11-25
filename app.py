import glob  # [ADDED - DAY 3] knowledge_base folder ki .txt files padhne ke liye
import json
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
from langchain_community.retrievers import BM25Retriever  # [ADDED - DAY 3] Keyword BM25 retriever
from sentence_transformers import CrossEncoder  # [ADDED - DAY 3] Cross-Encoder reranker

st.set_page_config(page_title="Mental Wellness AI Assistant", page_icon="🧠", layout="wide")  # [ADDED - DAY 3]
st.title("🧠 Mental Wellness Educational AI Assistant (Advanced RAG)")  # [ADDED - DAY 3]

KB_DIR = "knowledge_base"  # [ADDED - DAY 3]
DB_DIR = "chroma_db"
QA_MEMORY_FILE = "qa_memory.json"
HISTORY_FILE = "chat_history.json"


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource  # [ADDED - DAY 3] Cross-Encoder ko ek baar cache mein load karne ke liye
def load_reranker():  # [ADDED - DAY 3]
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # [ADDED - DAY 3]


def get_llm(model_name, temperature, top_p, max_tokens):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_tokens,
    )


embedding = load_embeddings()
reranker = load_reranker()  # [ADDED - DAY 3]


def load_kb_documents():  # [ADDED - DAY 3] 4 Mental Wellness .txt files ko category metadata ke saath load karta hai
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
    if vs._collection.count() == 0:  # [ADDED - DAY 3] Fresh DB mein Mental Wellness KB index karo
        kb_docs = load_kb_documents()
        if kb_docs:
            vs.add_documents(kb_docs)
    return vs


vectorstore = get_vectorstore()


def save_to_qa_memory(question, answer):
    qa_data = []
    if os.path.exists(QA_MEMORY_FILE) and os.path.getsize(QA_MEMORY_FILE) > 0:
        with open(QA_MEMORY_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

    if not any(item["question"].lower() == question.lower() for item in qa_data):
        qa_data.append({
            "question": question,
            "answer": answer,
            "embedding": embedding.embed_query(question),  # [ADDED - DAY 3] Terminal bot compatibility ke liye embedding bhi save
        })
        with open(QA_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)


def load_sessions():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):  # [ADDED - DAY 3] Agar terminal bot ne list save ki ho toh crash se bachne ke liye
                return data
    return {"session_1": []}


def save_sessions(sessions_data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions_data, f, indent=2, ensure_ascii=False)


if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()
if "current_session" not in st.session_state:
    st.session_state.current_session = "session_1"


with st.sidebar:
    st.header("🤖 LLM Settings")
    selected_model = st.selectbox("Model", ["qwen2.5:3b", "llama3.2:latest", "qwen2.5:1.5b"])  # [ADDED - DAY 3] qwen2.5:3b default
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9, step=0.1)
    max_tokens = st.slider("Max Tokens", 128, 1024, 450, step=64)
    st.divider()

    st.header("⚙️ RAG & Category Filter")  # [ADDED - DAY 3]
    rag_mode = st.radio(
        "RAG Pipeline",
        ["Hybrid + Rerank (Day 3)", "Semantic Only (Basic)"],  # [ADDED - DAY 3]
        horizontal=True,
    )
    # [ADDED - DAY 3] ChromaDB se dynamic categories nikalkar Metadata Filter dropdown banao
    raw_meta = vectorstore.get().get("metadatas", [])
    available_cats = sorted({m.get("category") for m in raw_meta if m and m.get("category")})
    selected_category = st.selectbox("Category Metadata Filter", ["All Categories"] + available_cats)  # [ADDED - DAY 3]

    top_k = st.slider("Final Top-K Chunks", 2, 6, 3)  # [ADDED - DAY 3]
    task_mode = st.selectbox("Task Mode", ["Question Answering", "Summarization", "Document Explanation"])
    length_mode = st.radio("Response Length", ["Short", "Medium", "Detailed"], horizontal=True)

    st.divider()

    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF ya TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk Size", 200, 1000, 500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, step=25)

    if st.button("Process & Add Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Documents process ho rahe hain..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            new_chunks = []
            for file in uploaded_files:
                if file.size == 0:
                    st.warning(f"⚠️ '{file.name}' empty file hai, skip kar diya gaya.")
                    continue

                suffix = os.path.splitext(file.name)[1].lower()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    if suffix == ".pdf":
                        loader = PyPDFLoader(tmp_path)
                    elif suffix == ".txt":
                        loader = TextLoader(tmp_path, encoding="utf-8")
                    else:
                        st.error(f"❌ Unsupported file format: {file.name}")
                        os.remove(tmp_path)
                        continue

                    docs = loader.load()
                    os.remove(tmp_path)

                    cat_name = os.path.splitext(file.name)[0].replace("_", " ").title()  # [ADDED - DAY 3]
                    valid_docs = []
                    for doc in docs:
                        cleaned = " ".join(doc.page_content.split())
                        if cleaned:
                            doc.page_content = cleaned
                            doc.metadata["source"] = file.name
                            doc.metadata["category"] = cat_name  # [ADDED - DAY 3]
                            valid_docs.append(doc)

                    if not valid_docs:
                        st.warning(f"⚠️ '{file.name}' mein koi readable text nahi mila.")
                        continue

                    new_chunks.extend(splitter.split_documents(valid_docs))
                except Exception as e:
                    st.error(f"❌ '{file.name}' process karne mein error: {e}")

            if new_chunks:
                vectorstore.add_documents(new_chunks)
                st.success(f"✅ {len(new_chunks)} chunks ChromaDB mein add ho gaye!")

    st.divider()
    st.header("💬 Chat Sessions")
    session_list = list(st.session_state.sessions.keys())
    selected_session = st.selectbox(
        "Active Session",
        session_list,
        index=session_list.index(st.session_state.current_session),
    )
    st.session_state.current_session = selected_session

    new_session_name = st.text_input("New Session Name", placeholder="e.g. session_2")
    if st.button("Create Session", use_container_width=True) and new_session_name.strip():
        name = new_session_name.strip()
        st.session_state.sessions.setdefault(name, [])
        st.session_state.current_session = name
        save_sessions(st.session_state.sessions)
        st.rerun()

    if st.button("Clear Current Chat", use_container_width=True):
        st.session_state.sessions[st.session_state.current_session] = []
        save_sessions(st.session_state.sessions)
        st.rerun()


llm = get_llm(selected_model, temperature, top_p, max_tokens)

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the Question into a clear standalone search query using Chat History so pronouns like 'it' or 'that' are resolved. "
    "Return ONLY the standalone query on one line.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Question: {question}\n"
    "Standalone Query:"
)

# [ADDED - DAY 3] Mental Wellness Safety & Grounding Rules
SAFETY_RULE = (
    "SAFETY RULE: Provide ONLY educational mental wellness tips strictly from Context. "
    "NEVER give medical diagnoses or prescriptions. Suggest a licensed doctor for severe distress."
)

TASK_PROMPTS = {
    "Question Answering": (
        "1. First, check Chat History for user's own details (name, age, etc.). If found, answer warmly.\n"
        "2. For mental wellness questions, answer strictly using only the provided Context.\n"
        "3. ONLY if a wellness topic is missing from Context, reply: 'Not found in the knowledge base.'\n"
        f"4. {SAFETY_RULE}"
    ),
    "Summarization": (
        "Summarize the retrieved Context into clear, accurate key points using ONLY the provided text. "
        f"If Context has no relevant notes, reply: 'Not found in the knowledge base.' {SAFETY_RULE}"
    ),
    "Document Explanation": (
        "Explain the concepts found in the retrieved Context in simple, clear terms strictly from the text. "
        f"If Context has no relevant notes, reply: 'Not found in the knowledge base.' {SAFETY_RULE}"
    ),
}

LENGTH_RULES = {
    "Short": "Keep your reply strictly short (1-2 sentences).",
    "Medium": "Provide a balanced answer in 3-4 concise bullet points (under 120 words).",
    "Detailed": "Provide a thorough, detailed explanation using structured bullet points.",
}


def format_history(session_id):
    history = st.session_state.sessions.get(session_id, [])[-10:]
    if not history:
        return "No previous conversation."
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def ask_stream(question, session_id):
    sessions = st.session_state.sessions
    text = question.lower().strip()
    greetings = ("hi", "hello", "hey", "namaste", "good morning", "good evening")

    if text in greetings:
        answer = "Hello! I am your Mental Wellness Educational Assistant. How can I help you today?"
        st.markdown(answer)
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions(sessions)
        return

    try:
        start_time = time.time()
        chat_history = format_history(session_id)

        # [ADDED - DAY 3] Sirf follow-up pronoun aane par hi query rewrite karo (fast & accurate)
        follow_up_words = ("it", "its", "that", "this", "they", "them", "those", "more", "explain")
        is_followup = any(w in text.split() for w in follow_up_words)
        if is_followup and len(sessions.get(session_id, [])) > 0:
            search_query = (rewrite_prompt | llm | StrOutputParser()).invoke(
                {"chat_history": chat_history, "question": question}
            ).strip()
            st.caption(f"🔎 **Rewritten Query:** `{search_query}`")
        else:
            search_query = question

        # --- [ADDED - DAY 3] Retrieval Pipeline ---
        if rag_mode == "Hybrid + Rerank (Day 3)":
            # Step 1: Hybrid Search (BM25 + Semantic RRF) + Metadata Filter
            raw = vectorstore.get()
            all_docs = []
            for i in range(len(raw["documents"])):
                t = raw["documents"][i]
                m = raw["metadatas"][i]
                if selected_category == "All Categories" or m.get("category") == selected_category:
                    all_docs.append(Document(page_content=t, metadata=m))

            bm25_ret = BM25Retriever.from_documents(all_docs)
            bm25_ret.k = 8

            search_kwargs = {"k": 8}
            if selected_category != "All Categories":
                search_kwargs["filter"] = {"category": selected_category}
            vec_ret = vectorstore.as_retriever(search_kwargs=search_kwargs)

            bm25_docs = bm25_ret.invoke(search_query)
            vec_docs = vec_ret.invoke(search_query)

            # Weighted Reciprocal Rank Fusion (40% BM25 + 60% Semantic)
            scores_map, doc_map = {}, {}
            for rank, d in enumerate(bm25_docs):
                doc_map[d.page_content] = d
                scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.4 / (rank + 60)
            for rank, d in enumerate(vec_docs):
                doc_map[d.page_content] = d
                scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.6 / (rank + 60)

            candidate_docs = [
                doc_map[k] for k, _ in sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
            ][:8]

            # Step 2: Cross-Encoder Reranking (Top-8 -> Top-K)
            if candidate_docs:
                pairs = [[search_query, d.page_content] for d in candidate_docs]
                rerank_scores = reranker.predict(pairs)
                ranked = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
                relevant_docs = [doc for _, doc in ranked[:top_k]]
            else:
                relevant_docs = []
        else:
            # Semantic Only (Basic) with optional Metadata Filter
            search_kwargs = {"k": top_k}
            if selected_category != "All Categories":
                search_kwargs["filter"] = {"category": selected_category}
            results = vectorstore.similarity_search_with_score(search_query, **search_kwargs)
            relevant_docs = [doc for doc, score in results if score <= 1.5]

        # [ADDED - DAY 3] Personal chat (name/age) detect karo taaki 'Not found' na bole
        personal_triggers = ("my name", "my age", "who am i", "what is my", "and age", "and name", "my brother")
        if any(text.startswith(p) for p in personal_triggers):
            context = "Personal user info shared or asked. Reply warmly in 1 sentence using Chat History only."
        else:
            context = (
                "\n\n".join(f"[{d.metadata.get('category', 'General')}] {d.page_content}" for d in relevant_docs)
                if relevant_docs
                else "No matching notes found."
            )

        dynamic_prompt = ChatPromptTemplate.from_template(
            f"You are a supportive AI Mental Wellness Educational Assistant. Follow these rules strictly:\n{TASK_PROMPTS[task_mode]}\n"
            f"{LENGTH_RULES[length_mode]}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "User Request: {question}\n"
            "Assistant:"
        )

        chain = dynamic_prompt | llm | StrOutputParser()

        answer = st.write_stream(
            chain.stream({"chat_history": chat_history, "context": context, "question": question})
        ).strip()

        elapsed = round(time.time() - start_time, 2)

        if "not found in the knowledge base" in answer.lower():
            footer = f"\n\n⚡ `{selected_model}` ({elapsed}s) | `{rag_mode}` | Filter: `{selected_category}`"
        else:
            save_to_qa_memory(question, answer)
            if relevant_docs and not any(text.startswith(p) for p in personal_triggers):
                sources_list = sorted({f"`{d.metadata.get('source', 'Unknown')}`" for d in relevant_docs})
                source_str = ", ".join(sources_list)
            else:
                source_str = "`Chat History`"
            footer = f"\n\n📚 **Sources:** {source_str} | 🏷️ `{selected_category}` | ⚡ `{selected_model}` ({elapsed}s) | `{rag_mode}`"

        st.markdown(footer)
        output = answer + footer

    except Exception as e:
        output = f"⚠️ LLM/API Error: Please check if Ollama is running (`ollama serve`). Details: {e}"
        st.error(output)

    sessions[session_id].append({"role": "user", "content": question})
    sessions[session_id].append({"role": "assistant", "content": output})
    save_sessions(sessions)


current_id = st.session_state.current_session
for msg in st.session_state.sessions.get(current_id, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question about Stress, Anxiety, Sleep, or Mindfulness..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        ask_stream(user_prompt, current_id)