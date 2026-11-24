import json
import os
import glob
import time  # [ADDED] Response speed (latency) measure karne ke liye
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
if os.path.exists(QA_MEMORY_FILE) and os.path.getsize(QA_MEMORY_FILE) > 0:  # [UPDATED] Empty JSON crash se bachne ke liye
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
    if os.path.exists(QA_MEMORY_FILE) and os.path.getsize(QA_MEMORY_FILE) > 0:  # [UPDATED] Safe empty check
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


# --- [COMMENTED OUT OLD FIXED LLM] ---
# llm = ChatOllama(
#     model="llama3.2",
#     temperature=0,
# )

# --- [ADDED] Dynamic Model & LLM Hyperparameter Controls ---
config = {  # [ADDED] Runtime par model, temp, top_p, rag_mode, task_mode aur length change karne ke liye
    "model": "llama3.2:latest",  # [ADDED] Default model (switchable to qwen2.5:3b / qwen2.5:1.5b)
    "temperature": 0.1,          # [ADDED] Randomness control (0.0 = deterministic, 1.0 = creative)
    "top_p": 0.9,                # [ADDED] Nucleus sampling threshold
    "max_tokens": 512,           # [ADDED] Maximum generated tokens limit
    "rag_mode": "advanced",      # [ADDED] "advanced" (Query Expansion + MMR) vs "basic"
    "task_mode": "qa",           # [ADDED] "qa" | "summary" | "explain"
    "length": "short",           # [ADDED] "short" | "medium" | "detailed"
}

def get_llm():  # [ADDED] Current config ke hisab se dynamic ChatOllama instance banata hai
    return ChatOllama(
        model=config["model"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        num_predict=config["max_tokens"],
    )


HISTORY_FILE = "chat_history.json"


def load_sessions():
    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:  # [UPDATED] Safe empty check
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


# --- [COMMENTED OUT OLD REWRITE PROMPT] ---
# rewrite_prompt = ChatPromptTemplate.from_template(
#     "Given the chat history and a follow-up question, rewrite the follow-up "
#     "question into a clear, standalone search query. Do NOT answer it. "
#     "If it is already standalone or unrelated, return it as is.\n\n"
#     "Chat History:\n{chat_history}\n\n"
#     "Follow-up Question: {question}\n"
#     "Standalone Query:"
# )

# --- [ADDED] Query Transformation + Query Expansion Prompt ---
rewrite_prompt = ChatPromptTemplate.from_template(  # [ADDED] Follow-up ko standalone banane + 2 related keywords add karne ke liye
    "Rewrite the user's question using Chat History so it is completely standalone, "
    "and append 2 concise related keywords to improve vector retrieval. "
    "Return ONLY the expanded query on one line.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Follow-up Question: {question}\n"
    "Expanded Standalone Query:"
)

# --- [COMMENTED OUT OLD SINGLE PROMPT] ---
# prompt = ChatPromptTemplate.from_template(
#     "You are a helpful, concise assistant. Follow these rules strictly:\n"
#     "1. First, check the Chat History for facts told by the user (like their name, age, preferences, etc.). If the answer is in Chat History, answer directly using that fact.\n"
#     "2. Second, check the Context (which includes PDF notes and previously saved Q&A). If the answer is in Context, answer using only that context.\n"
#     "3. ONLY if the answer is neither in Chat History nor in Context, reply: Not found in the notes.\n"
#     "4. Do NOT use outside world knowledge. Keep replies short (1-2 sentences).\n\n"
#     "Chat History:\n{chat_history}\n\n"
#     "Context:\n{context}\n\n"
#     "User Question: {question}\n"
#     "Assistant:"
# )

# --- [ADDED] Task-Specific Prompts (QA, Summarization, Explanation) & Length Modes ---
TASK_PROMPTS = {  # [ADDED] Alag-alag tasks ke liye specialized system instructions
    "qa": (
        "1. First, check Chat History for user facts. If found, answer directly.\n"
        "2. Second, check Context. Answer strictly using only the provided Context.\n"
        "3. ONLY if missing in both, reply: Not found in the notes. Do NOT use outside knowledge."
    ),
    "summary": (
        "Summarize the retrieved Context into clear, accurate key takeaways using ONLY the provided text. "
        "If Context has no relevant notes, reply: Not found in the notes."
    ),
    "explain": (
        "Explain the concepts from the retrieved Context clearly in simple terms with examples strictly grounded in the text. "
        "If Context has no relevant notes, reply: Not found in the notes."
    ),
}

LENGTH_RULES = {  # [ADDED] Short, Medium, aur Detailed output control
    "short": "Keep your reply strictly short (1-2 sentences).",
    "medium": "Provide a balanced 1-paragraph explanation (3-5 sentences).",
    "detailed": "Provide a thorough, detailed explanation using structured bullet points.",
}


def ask(question, session_id="session_1"):
    if session_id not in sessions:
        sessions[session_id] = []

    text = question.lower().strip()
    greetings = ("hi", "hello", "hey", "namaste", "good morning", "good evening")
    question_words = (
        "?", "what", "who", "where", "when", "why", "how",
        "kya", "kaun", "kab", "kaise", "kitna", "tell",
        "explain", "define", "list", "give", "summarize", "summary", "batao", "samjhao"  # [UPDATED] summarize keywords added
    )

    if text in greetings:
        answer = "Hello! How can I help you with your documents today?"
        print(f"\nBot: {answer}")  # [ADDED] Direct print for consistent terminal output
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions()
        return answer

    is_question = any(word in text for word in question_words)
    if not is_question and config["task_mode"] == "qa":  # [UPDATED] Sirf QA mode mein statements ko fact maan kar save karega
        answer = "Got it!"
        print(f"\nBot: {answer}")  # [ADDED] Direct print
        save_to_qa_memory(question, "User shared this information.")
        sessions[session_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        save_sessions()
        return answer

    try:  # [ADDED] Error handling agar Ollama band ho ya galat model select ho
        start_time = time.time()  # [ADDED] Generation speed track karne ke liye timer start
        llm = get_llm()  # [ADDED] Active model aur hyperparameters load karo
        chat_history = format_history(session_id)

        # --- [COMMENTED OUT OLD REWRITE CONDITION] ---
        # if sessions[session_id] and len(question.split()) <= 5:
        #     rewrite_chain = rewrite_prompt | llm | StrOutputParser()
        #     search_query = rewrite_chain.invoke(
        #         {"chat_history": chat_history, "question": question}
        #     ).strip()
        # else:
        #     search_query = question

        # --- [ADDED] Advanced RAG Query Transformation & Expansion ---
        if config["rag_mode"] == "advanced":  # [ADDED] Advanced RAG mein query ko expand aur contextualize karo
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            search_query = rewrite_chain.invoke(
                {"chat_history": chat_history, "question": question}
            ).strip()
            print(f"\n[Expanded Query]: {search_query}")  # [ADDED] Query transformation live dekhne ke liye
        else:
            search_query = question  # [ADDED] Basic RAG mein raw query use hoti hai

        # --- [COMMENTED OUT OLD SIMILARITY SEARCH ONLY] ---
        # results = vectorstore.similarity_search_with_score(search_query, k=5)
        # relevant_docs = [doc for doc, score in results if score <= 1.5]

        # --- [ADDED] MMR (Maximum Marginal Relevance) vs Basic Similarity Search ---
        if config["rag_mode"] == "advanced":  # [ADDED] MMR relevant + diverse chunks nikalta hai (duplicates hatata hai)
            relevant_docs = vectorstore.max_marginal_relevance_search(
                search_query, k=4, fetch_k=12, lambda_mult=0.6
            )
        else:  # [ADDED] Basic RAG comparison ke liye standard similarity search
            results = vectorstore.similarity_search_with_score(search_query, k=4)
            relevant_docs = [doc for doc, score in results if score <= 1.5]

        context = (
            "\n\n".join(d.page_content for d in relevant_docs)
            if relevant_docs
            else "No matching notes found."
        )

        # --- [ADDED] Dynamic Prompt (Task Mode + Length Mode) ---
        dynamic_prompt = ChatPromptTemplate.from_template(  # [ADDED] Active task aur length rule ke hisab se prompt banata hai
            f"You are a helpful AI assistant. Follow these rules:\n{TASK_PROMPTS[config['task_mode']]}\n"
            f"{LENGTH_RULES[config['length']]}\n\n"
            "Chat History:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "User Request: {question}\n"
            "Assistant:"
        )

        chain = dynamic_prompt | llm | StrOutputParser()

        # --- [COMMENTED OUT OLD NON-STREAMING INVOKE] ---
        # answer = chain.invoke(
        #     {
        #         "chat_history": chat_history,
        #         "context": context,
        #         "question": question,
        #     }
        # ).strip()

        # --- [ADDED] Token-by-Token Streaming Output ---
        print("\nBot: ", end="", flush=True)  # [ADDED] Streaming ke liye prefix print
        streamed_chunks = []  # [ADDED] Streamed tokens ko history mein save karne ke liye list
        for chunk in chain.stream(  # [ADDED] Token-by-token live response generation
            {
                "chat_history": chat_history,
                "context": context,
                "question": question,
            }
        ):
            print(chunk, end="", flush=True)  # [ADDED] Har token ko turant screen par print karo
            streamed_chunks.append(chunk)
        print()  # [ADDED] Stream khatam hone par newline
        answer = "".join(streamed_chunks).strip()  # [ADDED] Complete answer string assemble karo
        elapsed = round(time.time() - start_time, 2)  # [ADDED] Total response time in seconds

        if "not found in the notes" in answer.lower():
            output = answer
        else:
            save_to_qa_memory(question, answer)

            if relevant_docs:
                best_doc = relevant_docs[0]
                file_name = best_doc.metadata.get("source", "Unknown")
                page_num = best_doc.metadata.get("page", 0) + 1
                if file_name == QA_MEMORY_FILE:
                    source_info = f"📚 Source: {file_name}"
                else:
                    source_info = f"📚 Source: {file_name} (Page {page_num})"
            else:
                source_info = "📚 Source: Chat History"
            print(f"\n{source_info} | ⚡ {config['model']} ({elapsed}s) | Mode: {config['rag_mode'].upper()}")  # [ADDED] Source + Model + Speed print
            output = f"{answer}\n\n{source_info}"

    except Exception as e:  # [ADDED] Graceful error handling
        output = f"⚠️ Error: {e}"
        print(f"\n{output}")

    sessions[session_id].append({"role": "user", "content": question})
    sessions[session_id].append({"role": "assistant", "content": output})  # [UPDATED] Full output save
    save_sessions()

    return output


current_session = "session_1"
# --- [COMMENTED OUT OLD COMMANDS BANNER] ---
# print("\nCommands: /new <session_name> | /clear | /exit\n")
print("\nCommands: /new <name> | /clear | /model <name> | /temp <0-1> | /rag <basic|advanced> | /mode <qa|summary|explain> | /length <short|medium|detailed> | /exit\n")  # [ADDED] Updated help banner

while True:
    user_q = input(f"[{current_session} | {config['model']} | {config['rag_mode']}] You: ").strip()  # [UPDATED] Prompt mein active model aur RAG mode dikhega
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
    elif user_q.startswith("/model "):  # [ADDED] Switch Ollama model (e.g., /model qwen2.5:3b)
        config["model"] = user_q.split(" ", 1)[1].strip()
        print(f"Model switched to: {config['model']}")
    elif user_q.startswith("/temp "):  # [ADDED] Change temperature (e.g., /temp 0.7)
        config["temperature"] = float(user_q.split(" ", 1)[1].strip())
        print(f"Temperature set to: {config['temperature']}")
    elif user_q.startswith("/rag "):  # [ADDED] Switch between basic and advanced RAG
        val = user_q.split(" ", 1)[1].strip().lower()
        if val in ("basic", "advanced"):
            config["rag_mode"] = val
            print(f"RAG Mode set to: {config['rag_mode']}")
    elif user_q.startswith("/mode "):  # [ADDED] Switch between qa, summary, explain
        val = user_q.split(" ", 1)[1].strip().lower()
        if val in TASK_PROMPTS:
            config["task_mode"] = val
            print(f"Task Mode set to: {config['task_mode']}")
    elif user_q.startswith("/length "):  # [ADDED] Switch between short, medium, detailed
        val = user_q.split(" ", 1)[1].strip().lower()
        if val in LENGTH_RULES:
            config["length"] = val
            print(f"Response Length set to: {config['length']}")
    else:
        # --- [COMMENTED OUT OLD PRINT (kyunki ask() ab khud live stream print karta hai)] ---
        # print("\nBot:", ask(user_q, session_id=current_session))
        ask(user_q, session_id=current_session)  # [ADDED] Live streaming call
        print("-" * 40)