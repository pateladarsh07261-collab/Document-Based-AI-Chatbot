import json
import os
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
st.title("📚 Document-Based AI Assistant (RAG)")

DB_DIR = "chroma_db"
QA_MEMORY_FILE = "qa_memory.json"
HISTORY_FILE = "chat_history.json"


@st.cache_resource
def load_models():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3.2", temperature=0)
    return embedding, llm


embedding, llm = load_models()


def get_vectorstore():
    vs = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
    if vs._collection.count() == 0 and os.path.exists(QA_MEMORY_FILE):
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
    if os.path.exists(QA_MEMORY_FILE):
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
    if os.path.exists(HISTORY_FILE):
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


# Sidebar: PDF & TXT Upload + Chunking Controls + Sessions
with st.sidebar:
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


rewrite_prompt = ChatPromptTemplate.from_template(
    "Given the chat history and a follow-up question, rewrite the follow-up "
    "question into a clear, standalone search query. Do NOT answer it. "
    "If it is already standalone or unrelated, return it as is.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n"
    "Standalone Query:"
)

prompt = ChatPromptTemplate.from_template(
    "You are a helpful, concise assistant. Follow these rules strictly:\n"
    "1. First, check the Chat History for facts told by the user. If the answer is in Chat History, answer directly using that fact.\n"
    "2. Second, check the Context (uploaded documents and saved Q&A). If the answer is in Context, answer using only that context.\n"
    "3. ONLY if the answer is neither in Chat History nor in Context, reply: Not found in the notes.\n"
    "4. Do NOT use outside world knowledge. Keep replies short (1-2 sentences).\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "User Question: {question}\n"
    "Assistant:"
)


def format_history(session_id):
    history = st.session_state.sessions.get(session_id, [])[-10:]
    if not history:
        return "No previous conversation."
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def ask(question, session_id):
    sessions = st.session_state.sessions
    text = question.lower().strip()
    greetings = ("hi", "hello", "hey", "namaste", "good morning", "good evening")
    question_words = (
        "?", "what", "who", "where", "when", "why", "how",
        "kya", "kaun", "kab", "kaise", "kitna", "tell",
        "explain", "define", "list", "give", "batao", "samjhao"
    )

    if text in greetings:
        answer = "Hello! How can I help you with your documents today?"
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions(sessions)
        return answer

    is_question = any(word in text for word in question_words)
    if not is_question:
        answer = "Got it!"
        save_to_qa_memory(question, "User shared this information.")
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions(sessions)
        return answer

    try:
        chat_history = format_history(session_id)

        if sessions[session_id] and len(question.split()) <= 5:
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            search_query = rewrite_chain.invoke(
                {"chat_history": chat_history, "question": question}
            ).strip()
        else:
            search_query = question

        results = vectorstore.similarity_search_with_score(search_query, k=5)
        relevant_docs = [doc for doc, score in results if score <= 1.5]

        context = (
            "\n\n".join(d.page_content for d in relevant_docs)
            if relevant_docs
            else "No matching notes found."
        )

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke(
            {"chat_history": chat_history, "context": context, "question": question}
        ).strip()

        if "not found in the notes" in answer.lower():
            output = answer
        else:
            save_to_qa_memory(question, answer)
            if relevant_docs:
                best_doc = relevant_docs[0]
                file_name = best_doc.metadata.get("source", "Unknown")
                if file_name == QA_MEMORY_FILE or file_name.endswith(".txt"):
                    output = f"{answer}\n\n📚 **Source:** `{file_name}`"
                else:
                    page_num = best_doc.metadata.get("page", 0) + 1
                    output = f"{answer}\n\n📚 **Source:** `{file_name}` (Page {page_num})"
            else:
                output = answer

    except Exception as e:
        output = f"⚠️ LLM/API Error: Please check if Ollama is running (`ollama serve`). Details: {e}"

    sessions[session_id].append({"role": "user", "content": question})
    sessions[session_id].append({"role": "assistant", "content": output})
    save_sessions(sessions)
    return output


current_id = st.session_state.current_session
for msg in st.session_state.sessions.get(current_id, []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question from your uploaded PDF/TXT documents..."):
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            reply = ask(user_prompt, current_id)
            st.markdown(reply)