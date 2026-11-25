import os
import glob  # [ADDED - DAY 3] knowledge_base folder ki saari .txt files padhne ke liye
import json
import time
import numpy as np
# [OLD - DAY 2] from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader  # [ADDED - DAY 3] .txt files load karne ke liye
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document  # [ADDED - DAY 3] BM25 ke liye Document objects banane ke liye
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever  # [ADDED - DAY 3] Keyword-based BM25 retriever
from sentence_transformers import CrossEncoder  # [ADDED - DAY 3] 2-stage Cross-Encoder reranker

KB_DIR = "knowledge_base"  # [ADDED - DAY 3] Mental Wellness Knowledge Base folder
DB_DIR = "chroma_db"
HISTORY_FILE = "chat_history.json"
QA_MEMORY_FILE = "qa_memory.json"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # [ADDED - DAY 3] Cross-Encoder model load kiya


# --- 1. Knowledge Base Loader (Purane PDF Loader ki jagah) ---
# [OLD - DAY 2] def load_pdf():
# [OLD - DAY 2]     loader = PyPDFLoader("Function_Interview_ 5.pdf")
# [OLD - DAY 2]     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# [OLD - DAY 2]     return splitter.split_documents(loader.load())

def load_kb_documents():  # [ADDED - DAY 3] 4 Mental Wellness .txt files ko category metadata ke saath load karta hai
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # [ADDED - DAY 3]
    chunks = []  # [ADDED - DAY 3]
    for path in glob.glob(os.path.join(KB_DIR, "*.txt")):  # [ADDED - DAY 3]
        fname = os.path.basename(path)  # [ADDED - DAY 3]
        category = fname.replace(".txt", "").replace("_", " ").title()  # [ADDED - DAY 3] e.g. Stress Management
        loader = TextLoader(path, encoding="utf-8")  # [ADDED - DAY 3]
        for doc in loader.load():  # [ADDED - DAY 3]
            doc.page_content = " ".join(doc.page_content.split())  # [ADDED - DAY 3]
            doc.metadata = {"source": fname, "category": category, "page": 0}  # [ADDED - DAY 3]
            chunks.extend(splitter.split_documents([doc]))  # [ADDED - DAY 3]
    return chunks  # [ADDED - DAY 3]


vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
if vectorstore._collection.count() == 0:
    # [OLD - DAY 2] vectorstore.add_documents(load_pdf())
    vectorstore.add_documents(load_kb_documents())  # [ADDED - DAY 3] Fresh DB mein Mental Wellness KB index kiya


# --- 2. Persistent Chat History & Q&A Semantic Memory (Same as Day 2) ---
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


chat_history = load_json(HISTORY_FILE, [])
if not isinstance(chat_history, list): # chack karta h chat history me list ya nhi 
    chat_history = []

qa_memory = load_json(QA_MEMORY_FILE, [])
updated = False
for item in qa_memory:
    if "embedding" not in item:
        item["embedding"] = embedding.embed_query(item["question"])
        updated = True
if updated:
    save_json(QA_MEMORY_FILE, qa_memory)

def search_qa_memory(query, threshold=0.82):
    if not qa_memory:
        return None
    q_vec = np.array(embedding.embed_query(query))
    best_score, best_ans = -1.0, None
    for item in qa_memory:
        m_vec = np.array(item["embedding"])
        sim = float(np.dot(q_vec, m_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(m_vec) + 1e-9))
        if sim > best_score:
            best_score, best_ans = sim, item["answer"]
    return best_ans if best_score >= threshold else None
#

def save_qa_memory(question, answer):
    qa_memory.append({
        "question": question,
        "answer": answer,
        "embedding": embedding.embed_query(question),
    })
    save_json(QA_MEMORY_FILE, qa_memory)


# --- 3. LLM & Prompts ---
llm = ChatOllama(model="qwen2.5:3b", temperature=0.1, top_p=0.9, num_predict=450)
selected_category = "All Categories"  # [ADDED - DAY 3] Context-aware metadata filter ke liye default value

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the Question into a clear standalone search query using Chat History so pronouns like 'it' or 'that' are resolved. "
    "Return ONLY the standalone query on one line.\n\nChat History:\n{chat_history}\n\nQuestion: {question}\nStandalone Query:"
)

qa_prompt = ChatPromptTemplate.from_template(
    "You are a supportive AI Mental Wellness Educational Assistant.\n"  # [ADDED - DAY 3]
    "Rules:\n"
    "1. For greetings, introductions, or questions about the user's own details (name, age, etc.), answer warmly in 1 sentence using Chat History ONLY—never say 'Not found in the knowledge base' for personal chat.\n"  # [ADDED - DAY 3]
    "2. For mental wellness questions, answer strictly using Context in 3-4 concise bullet points (under 120 words). Only if a wellness topic is missing from Context, reply: 'Not found in the knowledge base.'\n"
    "3. SAFETY RULE: Provide ONLY educational mental wellness tips from Context. NEVER give medical diagnoses or prescriptions. Suggest a licensed doctor for severe distress.\n\n"  # [ADDED - DAY 3]
    "Chat History:\n{chat_history}\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

##
# --- 4. Main Chat Loop ---
while True:
    user_q = input("You: ").strip()
    if not user_q:
        continue
    if user_q.lower() in ("exit", "quit"):
        break

    # [ADDED - DAY 3] Category filter command (e.g. /cat Stress Management ya /cat all)
    if user_q.startswith("/cat "):  # [ADDED - DAY 3]
        val = user_q.split(" ", 1)[1].strip()  # [ADDED - DAY 3]
        selected_category = "All Categories" if val.lower() == "all" else val  # [ADDED - DAY 3]
        print(f"⚙️ Category Filter: {selected_category}\n")  # [ADDED - DAY 3]
        continue  # [ADDED - DAY 3]

    t0 = time.time()

    # Check Q&A Embedding Memory first (Same as Day 2)
    cached_ans = search_qa_memory(user_q)
    if cached_ans:
        print(f"⚡ [Memory] AI: {cached_ans}\n⏱️ ({round(time.time() - t0, 2)}s)\n")
        chat_history.extend([f"User: {user_q}", f"Assistant: {cached_ans}"])
        save_json(HISTORY_FILE, chat_history)
        continue

    # [OLD] hist_text = "\n".join(chat_history[-6:]) if chat_history else "None"
    # [ADDED - MEMORY FIX] Recent chat history + Q&A Embedding Memory se top-3 relevant past Q&A nikalo
    recent_hist = chat_history[-6:]  # [ADDED - MEMORY FIX]
    mem_hits = []  # [ADDED - MEMORY FIX]
    if len(qa_memory) > 0:
        new_vec = embedding.embed_query(user_q)
        scored_mem = []

        # 1. Har purane Q&A par loop chalao aur similarity score nikalo
        for item in qa_memory:
            old_vec = item["embedding"]

            dot_product = 0
            new_len = 0
            old_len = 0

            for i in range(len(new_vec)):
                dot_product = dot_product + (new_vec[i] * old_vec[i])
                new_len = new_len + (new_vec[i] * new_vec[i])
                old_len = old_len + (old_vec[i] * old_vec[i])

            sim = dot_product / ((new_len ** 0.5) * (old_len ** 0.5) + 1e-9)

            # 2. Agar 35% (0.35) ya usse zyada match hai to list mein daalo
            if sim >= 0.35:
                past_text = "Past User: " + item["question"] + "\nPast Assistant: " + item["answer"]
                scored_mem.append((sim, past_text))

        # 3. Sabse bade score ko upar lane ke liye sort karo
        scored_mem.sort(reverse=True)

        # 4. Top-3 items mein se sirf text nikalo (simple loop se)
        mem_hits = []
        top_three = scored_mem[:3]
        for score, txt in top_three:
            mem_hits.append(txt)

    # 5. Purani yaadein aur abhi ke last 3 turns ko jodo
    all_history = mem_hits + recent_hist
    if len(all_history) > 0:
        hist_text = "\n".join(all_history)
    else:
        hist_text = "None"
        
    # Sirf pronoun aane par hi history se query rewrite karo  # [ADDED - DAY 3]
    follow_up_words = ("it", "its", "that", "this", "they", "them", "those", "more", "explain")  # [ADDED - DAY 3]
    is_followup = any(w in user_q.lower().split() for w in follow_up_words)  # [ADDED - DAY 3]
    if is_followup and chat_history:  # [ADDED - DAY 3]
        search_query = (rewrite_prompt | llm | StrOutputParser()).invoke({"chat_history": hist_text, "question": user_q}).strip()
    else:  # [ADDED - DAY 3]
        search_query = user_q  # [ADDED - DAY 3]

    # [OLD - DAY 2] results = vectorstore.similarity_search_with_score(search_query, k=4)
    # [OLD - DAY 2] relevant_docs = [doc for doc, score in results if score <= 1.5]

    # --- [ADDED - DAY 3] Step 1: Hybrid Search (BM25 Keyword + Semantic Vector) + Metadata Filter ---
    raw = vectorstore.get()  # [ADDED - DAY 3] ChromaDB se saare chunks nikale
    all_docs = [  # [ADDED - DAY 3]
        Document(page_content=t, metadata=m)  # [ADDED - DAY 3]
        for t, m in zip(raw["documents"], raw["metadatas"])  # [ADDED - DAY 3]
        if selected_category == "All Categories" or m.get("category") == selected_category  # [ADDED - DAY 3]
    ]  # [ADDED - DAY 3]

    bm25_ret = BM25Retriever.from_documents(all_docs)  # [ADDED - DAY 3] BM25 keyword retriever
    bm25_ret.k = 8  # [ADDED - DAY 3]
    search_kwargs = {"k": 8}  # [ADDED - DAY 3]
    if selected_category != "All Categories":  # [ADDED - DAY 3]
        search_kwargs["filter"] = {"category": selected_category}  # [ADDED - DAY 3] Metadata filtering
    vec_ret = vectorstore.as_retriever(search_kwargs=search_kwargs)  # [ADDED - DAY 3] Semantic retriever

    bm25_docs = bm25_ret.invoke(search_query)  # [ADDED - DAY 3]
    vec_docs = vec_ret.invoke(search_query)  # [ADDED - DAY 3]

    # Weighted Reciprocal Rank Fusion (40% BM25 + 60% Semantic)  # [ADDED - DAY 3]
    scores_map, doc_map = {}, {}  # [ADDED - DAY 3]
    for rank, d in enumerate(bm25_docs):  # [ADDED - DAY 3]
        doc_map[d.page_content] = d  # [ADDED - DAY 3]
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.4 / (rank + 60)  # [ADDED - DAY 3]
    for rank, d in enumerate(vec_docs):  # [ADDED - DAY 3]
        doc_map[d.page_content] = d  # [ADDED - DAY 3]
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.6 / (rank + 60)  # [ADDED - DAY 3]
    candidate_docs = [doc_map[k] for k, _ in sorted(scores_map.items(), key=lambda x: x[1], reverse=True)][:8]  # [ADDED - DAY 3]

    # --- [ADDED - DAY 3] Step 2: Cross-Encoder Reranking (Top-8 se Top-3 best chunks) ---
    pairs = [[search_query, d.page_content] for d in candidate_docs]  # [ADDED - DAY 3]
    rerank_scores = reranker.predict(pairs)  # [ADDED - DAY 3] Har chunk ko query ke saath score diya
    ranked = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)  # [ADDED - DAY 3]
    relevant_docs = [doc for _, doc in ranked[:3]]  # [ADDED - DAY 3] Top-3 best chunks LLM ko bheje

    # [ADDED - DAY 3] Personal chat / intro detect karo taaki 'Not found in KB' na bole
    # [OLD] personal_triggers = ("hi", "hello", "hey", "my name", "my age", "who am i", "what is my")

    personal_triggers = ("hi", "hello", "hey", "my name", "my age", "who am i", "what is my", "and age", "and name")  # [ADDED - MEMORY FIX]
    if any(user_q.lower().startswith(p) for p in personal_triggers):  # [ADDED - DAY 3]
        context = "Personal greeting/introduction. Reply warmly in 1 sentence using Chat History only."  # [ADDED - DAY 3]
    else:  # [ADDED - DAY 3]
        context = "\n\n".join(f"[{d.metadata.get('category')}] {d.page_content}" for d in relevant_docs) if relevant_docs else "No matching notes found."  # [ADDED - DAY 3]

    print(f"🔍 Search Query: {search_query}")
    print(f"📄 Sources: {[d.metadata.get('source') for d in relevant_docs]}")
    print("AI: ", end="", flush=True)

    # [ADDED - DAY 3] Personal chat ho toh seedha chat prompt use karo (taaki 'Not found in KB' rule na chale)
    if any(user_q.lower().startswith(p) for p in personal_triggers):  # [ADDED - DAY 3]
        chat_only_prompt = ChatPromptTemplate.from_template(  # [ADDED - DAY 3]
            "You are a friendly assistant. Reply warmly in 1 short sentence using Chat History.\nChat History:\n{chat_history}\nUser: {question}\nAssistant:"  # [ADDED - DAY 3]
        )  # [ADDED - DAY 3]
        chain = chat_only_prompt | llm | StrOutputParser()  # [ADDED - DAY 3]
    else:  # [ADDED - DAY 3]
        chain = qa_prompt | llm | StrOutputParser()  # [ADDED - DAY 3]


    full_ans = []
    for chunk in chain.stream({"chat_history": hist_text, "context": context, "question": user_q}):
        print(chunk, end="", flush=True)
        full_ans.append(chunk)

    ans_text = "".join(full_ans).strip()
    print(f"\n⏱️ ({round(time.time() - t0, 2)}s)\n")

    # Save to both Chat History JSON & Q&A Embedding Memory JSON (Same as Day 2)
    chat_history.extend([f"User: {user_q}", f"Assistant: {ans_text}"])
    save_json(HISTORY_FILE, chat_history)
    # [OLD] if "Not found in the knowledge base" not in ans_text and not any(user_q.lower().startswith(p) for p in personal_triggers):
    if "Not found in the knowledge base" not in ans_text:  # [ADDED - MEMORY FIX] Personal facts (name/age) bhi embedding memory mein save honge
        save_qa_memory(user_q, ans_text)