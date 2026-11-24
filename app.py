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

st.set_page_config(page_title="Document-Based AI Assistant", page_icon="📚", layout="wide")
st.title("📚 Document-Based AI Assistant (Advanced RAG)") 

DB_DIR = "chroma_db"
QA_MEMORY_FILE = "qa_memory.json"
HISTORY_FILE = "chat_history.json"


@st.cache_resource
def load_embeddings():  
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm(model_name, temperature, top_p, max_tokens):  
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_tokens,
    )


embedding = load_embeddings()


def get_vectorstore():
    vs = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
    if (
        vs._collection.count() == 0
        and os.path.exists(QA_MEMORY_FILE)
        and os.path.getsize(QA_MEMORY_FILE) > 0  
    ):
        with open(QA_MEMORY_FILE, "r", encoding="utf-8") as f:
            saved_qa = json.load(f)
            docs = [
                Document(
                    page_content=f"Question: {item['question']}\nAnswer: {item['answer']}",
                    metadata={"source": QA_MEMORY_FILE, "page": 0},
                )
                for item in saved_qa
            ]
            if docs:
                vs.add_documents(docs)
    return vs


vectorstore = get_vectorstore()


def save_to_qa_memory(question, answer):
    qa_data = []
    if os.path.exists(QA_MEMORY_FILE) and os.path.getsize(QA_MEMORY_FILE) > 0:  
        with open(QA_MEMORY_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

    if not any(item["question"].lower() == question.lower() for item in qa_data):
        qa_data.append({"question": question, "answer": answer})
        with open(QA_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)

        vectorstore.add_texts(
            texts=[f"Question: {question}\nAnswer: {answer}"],
            metadatas=[{"source": QA_MEMORY_FILE, "page": 0}],
        )


def load_sessions():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:  
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
    selected_model = st.selectbox("Model", ["llama3.2:latest", "qwen2.5:3b", "qwen2.5:1.5b"])  
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1)  
    top_p = st.slider("Top-P", 0.1, 1.0, 0.9, step=0.1)  
    max_tokens = st.slider("Max Tokens", 128, 1024, 512, step=64)  
    st.divider()  

    
    st.header("⚙️ RAG & Prompt Mode")  
    rag_mode = st.radio("RAG Pipeline", ["Advanced RAG", "Basic RAG"], horizontal=True)  
    top_k = st.slider("Top-K Chunks", 2, 8, 4)  
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

                    valid_docs = []
                    for doc in docs:
                        cleaned = " ".join(doc.page_content.split())
                        if cleaned:
                            doc.page_content = cleaned
                            doc.metadata["source"] = file.name
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
    "Rewrite the user's question using Chat History so it is completely standalone, "
    "and append 2 concise related keywords to improve document retrieval. "  
    "Return ONLY the expanded query on one line.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n"
    "Expanded Query:"
)

TASK_PROMPTS = {
    "Question Answering": (
        "1. First, check Chat History for facts told by the user. If found, answer directly.\n"
        "2. Second, check Context. Answer strictly using only the provided Context.\n"
        "3. ONLY if missing in both, reply: Not found in the notes. Do NOT use outside world knowledge."
    ),
    "Summarization": (
        "Summarize the retrieved Context into clear, accurate key points using ONLY the provided text. "
        "If Context has no relevant notes, reply: Not found in the notes."
    ),
    "Document Explanation": (
        "Explain the concepts found in the retrieved Context in simple, clear terms with examples strictly from the text. "
        "If Context has no relevant notes, reply: Not found in the notes."
    ),
}

LENGTH_RULES = {
    "Short": "Keep your reply strictly short (1-2 sentences).",
    "Medium": "Provide a balanced 1-paragraph answer (3-5 sentences).",
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
    question_words = (
        "?", "what", "who", "where", "when", "why", "how",
        "kya", "kaun", "kab", "kaise", "kitna", "tell",
        "explain", "define", "list", "give", "summarize", "summary", "batao", "samjhao"  
    )

    if text in greetings:
        answer = "Hello! How can I help you with your documents today?"
        st.markdown(answer)
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions(sessions)
        return

    is_question = any(word in text for word in question_words)
    if not is_question and task_mode == "Question Answering":  
        answer = "Got it!"
        st.markdown(answer)
        save_to_qa_memory(question, "User shared this information.")
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions(sessions)
        return

    try:
        start_time = time.time()  
        chat_history = format_history(session_id)

        
        if rag_mode == "Advanced RAG":
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            search_query = rewrite_chain.invoke(
                {"chat_history": chat_history, "question": question}
            ).strip()
            st.caption(f"🔎 **Expanded Query:** `{search_query}`")  
        else:
            search_query = question

        if rag_mode == "Advanced RAG":
            relevant_docs = vectorstore.max_marginal_relevance_search(
                search_query, k=top_k, fetch_k=top_k * 3, lambda_mult=0.6
            )
        else:
            results = vectorstore.similarity_search_with_score(search_query, k=top_k)
            relevant_docs = [doc for doc, score in results if score <= 1.5]

        context = (
            "\n\n".join(d.page_content for d in relevant_docs)
            if relevant_docs
            else "No matching notes found."
        )

        dynamic_prompt = ChatPromptTemplate.from_template(
            f"You are a helpful AI assistant. Follow these rules strictly:\n{TASK_PROMPTS[task_mode]}\n"
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

        if "not found in the notes" in answer.lower():
            footer = f"\n\n⚡ `{selected_model}` ({elapsed}s) | `{rag_mode}`"
        else:
            save_to_qa_memory(question, answer)
            if relevant_docs:
                best_doc = relevant_docs[0]
                file_name = best_doc.metadata.get("source", "Unknown")
                if file_name == QA_MEMORY_FILE or file_name.endswith(".txt"):
                    source_str = f"`{file_name}`"
                else:
                    page_num = best_doc.metadata.get("page", 0) + 1
                    source_str = f"`{file_name}` (Page {page_num})"
            else:
                source_str = "`Chat History`"
            footer = f"\n\n📚 **Source:** {source_str} | ⚡ `{selected_model}` ({elapsed}s) | `{rag_mode}`"

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

if user_prompt := st.chat_input("Ask a question from your uploaded PDF/TXT documents..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        ask_stream(user_prompt, current_id)  