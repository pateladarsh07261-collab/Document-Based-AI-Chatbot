import json
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


all_chunks = []
pdf_files = glob.glob("*.pdf")  
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())
        doc.metadata["source"] = pdf  # Kis PDF ka chunk hai, ye save rahega
    all_chunks.extend(splitter.split_documents(docs))

QA_MEMORY_FILE = "qa_memory.json"
if os.path.exists(QA_MEMORY_FILE):
    with open(QA_MEMORY_FILE, "r", encoding="utf-8") as f:
        saved_qa = json.load(f)
        for item in saved_qa:
            qa_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            all_chunks.append(
                Document(page_content=qa_text, metadata={"source": QA_MEMORY_FILE, "page": 0})
            )

print(f"Loaded {len(pdf_files)} PDFs | Total chunks: {len(all_chunks)}")


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_DIR = "chroma_db"
if os.path.exists(DB_DIR):
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding,
    )
    print("Old Db Loaded")
else:
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding,
        persist_directory=DB_DIR,
    )
    print("New DB Added")


def save_to_qa_memory(question, answer):
    """Verified Q&A aur User Facts ko JSON aur ChromaDB dono mein save karega."""
    qa_data = []
    if os.path.exists(QA_MEMORY_FILE):
        with open(QA_MEMORY_FILE, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

    # Duplicate entry se bachne ke liye check
    if not any(item["question"].lower() == question.lower() for item in qa_data):
        qa_data.append({"question": question, "answer": answer})
        with open(QA_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)

        vectorstore.add_texts(
            texts=[f"Question: {question}\nAnswer: {answer}"],
            metadatas=[{"source": QA_MEMORY_FILE, "page": 0}],
        )


llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)


HISTORY_FILE = "chat_history.json"


def load_sessions():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"session_1": []}


def save_sessions():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


sessions = load_sessions()


def format_history(session_id):
    history = sessions.get(session_id, [])[-10:]
    if not history:
        return "No previous conversation."
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def clear_history(session_id):
    sessions[session_id] = []
    save_sessions()
    print(f"Chat history cleared for '{session_id}'!")


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
    "1. First, check the Chat History for facts told by the user (like their name, age, preferences, etc.). If the answer is in Chat History, answer directly using that fact.\n"
    "2. Second, check the Context (which includes PDF notes and previously saved Q&A). If the answer is in Context, answer using only that context.\n"
    "3. ONLY if the answer is neither in Chat History nor in Context, reply: Not found in the notes.\n"
    "4. Do NOT use outside world knowledge. Keep replies short (1-2 sentences).\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "User Question: {question}\n"
    "Assistant:"
)


def ask(question, session_id="session_1"):
    if session_id not in sessions:
        sessions[session_id] = []

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
        save_sessions()
        return answer

    is_question = any(word in text for word in question_words)
    if not is_question:
        answer = "Got it!"
        save_to_qa_memory(question, "User shared this information.")
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions()
        return answer

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
        {
            "chat_history": chat_history,
            "context": context,
            "question": question,
        }
    ).strip()

    if "not found in the notes" in answer.lower():
        output = answer
    else:
        save_to_qa_memory(question, answer)

        if relevant_docs:
            best_doc = relevant_docs[0]
            file_name = best_doc.metadata.get("source", "Unknown")
            page_num = best_doc.metadata.get("page", 0) + 1
            if file_name == QA_MEMORY_FILE:
                output = f"{answer}\n\n📚 Source: {file_name}"
            else:
                output = f"{answer}\n\n📚 Source: {file_name} (Page {page_num})"
        else:
            output = answer

    sessions[session_id].append({"role": "user", "content": question})
    sessions[session_id].append({"role": "assistant", "content": answer})
    save_sessions()

    return output


current_session = "session_1"
print("\nCommands: /new <session_name> | /clear | /exit\n")

while True:
    user_q = input(f"[{current_session}] You: ").strip()
    if not user_q:
        continue

    if user_q == "/exit":
        break
    elif user_q == "/clear":
        clear_history(current_session)
    elif user_q.startswith("/new "):
        current_session = user_q.split(" ", 1)[1].strip()
        sessions.setdefault(current_session, [])
        save_sessions()
        print(f"Switched to '{current_session}'")
    else:
        print("\nBot:", ask(user_q, session_id=current_session))
        print("-" * 40)