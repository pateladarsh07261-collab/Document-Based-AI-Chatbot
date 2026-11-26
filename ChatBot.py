import os
import glob
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import memory_manager as mm  # [DAY 4] SQLite Short-Term, Long-Term & User Profile Manager

KB_DIR = "knowledge_base"
DB_DIR = "chroma_db"

# Initialize SQLite Database & 3 Dummy Profiles (user_101, user_102, user_103)
mm.init_db()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


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


vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
if vectorstore._collection.count() == 0:
    vectorstore.add_documents(load_kb_documents())


# --- 2. LLM & Prompts (Day 3 + Day 4) ---
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.25,
    top_p=0.9,
    repeat_penalty=1.25,  # Repetition loop rokne ke liye
    num_predict=220,      # Fast & concise output ke liye
)
selected_category = "All Categories"

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
    "- Name: {user_name} ({age_group})\n"
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
    "1. For greetings, introductions, or questions about user profile/memories, reply warmly using User Profile, Memories, and Chat History—never say 'Not found in the knowledge base.'\n"
    "2. If the user asks about past feelings or why they were stressed, directly state the exact reason from RETRIEVED LONG-TERM MEMORIES first, then give 3 short bullet points from Knowledge Base Context.\n"
    "3. For mental wellness questions, answer strictly using Knowledge Base Context in 3-4 concise bullet points (under 80 words total). Only if an unrelated factual/medical topic is completely missing from Context, reply: 'Not found in the knowledge base.'\n"
    "4. SAFETY RULE: Provide educational wellness tips only. Never diagnose or prescribe medication.\n\n"
    "Question: {question}\n"
    "Answer:"
)


# --- 3. Helper Functions for Day 4 Memory & Auto-Summarization ---
def maybe_summarize(user_id, session_id, profile):
    msgs = mm.get_session_messages(user_id, session_id)
    if len(msgs) >= 8 and len(msgs) % 4 == 0:
        recent_text = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[-8:])
        new_sum = (summary_prompt | llm | StrOutputParser()).invoke(
            {"old_summary": profile["conversation_summary"], "recent_chat": recent_text}
        ).strip()
        mm.update_summary(user_id, new_sum)
        print("📝 [Auto-Summary Updated]")


def extract_and_store_fact(user_id, question, profile):
    if not profile["memory_enabled"]:
        return
    personal_markers = (
        "my name", "i am", "i'm", "i feel", "i have", "my exam", "my work",
        "mera naam", "mujhe", "main ", "i like", "i prefer", "stress about", "trouble sleeping"
    )
    if any(m in question.lower() for m in personal_markers) and len(question.split()) >= 3:
        mm.save_long_term_memory(user_id, f"User shared: {question.strip()}", embedding)


# --- 4. User Selection & Terminal Commands ---
users = mm.get_all_users()
print("\n=== 🧠 Humaeli Terminal Chatbot (Day 3 Hybrid RAG + Day 4 Memory & Personalization) ===")
print("Available Users:")
for u in users:
    print(f"  - {u['user_id']} ({u['name']})")

current_user_id = input("Enter User ID (default user_101): ").strip() or "user_101"
if not mm.get_profile(current_user_id):
    print("⚠️ Invalid User ID, defaulting to user_101")
    current_user_id = "user_101"

current_session_id = "session_1"
profile = mm.get_profile(current_user_id)
print(f"\n✅ Logged in as: {profile['name']} | Lang: {profile['preferred_language']} | Memory: {'ON' if profile['memory_enabled'] else 'OFF'}")
print("Commands: /user <id> | /session <name> | /memories | /delmem <id> | /mem on|off | /clear | /cat <name> | exit\n")


# --- 5. Main Chat Loop ---
while True:
    user_q = input(f"[{current_user_id} | {current_session_id}] You: ").strip()
    if not user_q:
        continue
    if user_q.lower() in ("exit", "quit"):
        break

    # [DAY 3] Category Metadata Filter
    if user_q.startswith("/cat "):
        val = user_q.split(" ", 1)[1].strip()
        selected_category = "All Categories" if val.lower() == "all" else val
        print(f"⚙️ Category Filter: {selected_category}\n")
        continue

    # [DAY 4] Switch User Profile (Strict User Isolation)
    if user_q.startswith("/user "):
        uid = user_q.split(" ", 1)[1].strip()
        if mm.get_profile(uid):
            current_user_id = uid
            current_session_id = "session_1"
            profile = mm.get_profile(current_user_id)
            print(f"👤 Switched to {profile['name']} ({profile['preferred_language']}) | Session: {current_session_id}\n")
        else:
            print("❌ User not found. Use user_101, user_102, or user_103.\n")
        continue

    # [DAY 4] Switch or Create Separate Session
    if user_q.startswith("/session "):
        current_session_id = user_q.split(" ", 1)[1].strip()
        print(f"💬 Active Session: {current_session_id}\n")
        continue

    # [DAY 4] View Saved Long-Term Memories
    if user_q == "/memories":
        mems = mm.get_all_memories(current_user_id)
        if not mems:
            print("🧠 No long-term memories saved yet.\n")
        else:
            print("🧠 Saved Long-Term Memories:")
            for m in mems:
                print(f"  [{m['id']}] {m['fact']} ({m['created_at']})")
            print()
        continue

    # [DAY 4] Delete Individual Memory by ID
    if user_q.startswith("/delmem "):
        try:
            mid = int(user_q.split(" ", 1)[1].strip())
            mm.delete_memory_by_id(current_user_id, mid)
            print(f"🗑️ Deleted memory ID {mid} (if it belonged to {current_user_id}).\n")
        except ValueError:
            print("❌ Please provide a valid numeric ID, e.g., /delmem 1\n")
        continue

    # [DAY 4] Enable / Disable Long-Term Memory
    if user_q.startswith("/mem "):
        state = user_q.split(" ", 1)[1].strip().lower()
        enabled = 1 if state == "on" else 0
        p = mm.get_profile(current_user_id)
        mm.update_profile(
            current_user_id, p["age_group"], p["preferred_language"],
            p["interests"], p["communication_style"], p["past_concerns"], enabled
        )
        print(f"🔒 Long-Term Memory is now {'ON' if enabled else 'OFF'}.\n")
        continue

    # [DAY 4] Clear Current Session Chat History
    if user_q == "/clear":
        mm.clear_session_history(current_user_id, current_session_id)
        print(f"🧹 Cleared chat history for {current_user_id} -> {current_session_id}\n")
        continue

    t0 = time.time()
    profile = mm.get_profile(current_user_id)

    # [DAY 4] Short-Term Session History (last 6 messages)
    session_msgs = mm.get_session_messages(current_user_id, current_session_id)
    hist_text = "\n".join(f"{m['role']}: {m['content']}" for m in session_msgs[-6:]) if session_msgs else "None"

    # [DAY 4] Long-Term Cross-Session Memory Retrieval (Cosine Similarity >= 0.28)
    lt_memories = mm.search_long_term_memories(current_user_id, user_q, embedding, min_score=0.28, top_k=3)
    lt_mem_text = "\n".join(f"- {m}" for m in lt_memories) if lt_memories else "None"

    # [DAY 3 + DAY 4] Context-Aware Query Rewriting for Follow-Up Questions
    follow_up_words = ("it", "its", "that", "this", "they", "them", "those", "more", "explain", "kaise", "kya")
    is_followup = any(w in user_q.lower().split() for w in follow_up_words)
    if is_followup and (session_msgs or lt_memories):
        combined_hist = f"Memories:\n{lt_mem_text}\nRecent Chat:\n{hist_text}"
        search_query = (rewrite_prompt | llm | StrOutputParser()).invoke(
            {"chat_history": combined_hist, "question": user_q}
        ).strip()
    else:
        search_query = user_q

    # --- [DAY 3] Step 1: Hybrid Search (BM25 + Semantic RRF) + Metadata Filter ---
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

    # --- [DAY 3] Step 2: Cross-Encoder Reranking (Top-8 -> Top-3) ---
    pairs = [[search_query, d.page_content] for d in candidate_docs]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True)
    relevant_docs = [doc for _, doc in ranked[:3]]

    personal_triggers = ("hi", "hello", "hey", "my name", "my age", "who am i", "what is my")
    if any(user_q.lower().startswith(p) for p in personal_triggers):
        context = "Personal greeting/introduction. Reply warmly using User Profile, Memories, and Chat History."
    else:
        context = "\n\n".join(f"[{d.metadata.get('category')}] {d.page_content}" for d in relevant_docs) if relevant_docs else "No matching notes found."

    print(f"🔍 Search Query: {search_query} | 🧠 Memories Used: {len(lt_memories)}")
    print(f"📄 Sources: {[d.metadata.get('source') for d in relevant_docs]}")
    print("Humaeli: ", end="", flush=True)

    chain = qa_prompt | llm | StrOutputParser()
    full_ans = []
    for chunk in chain.stream({
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
        "question": user_q,
    }):
        print(chunk, end="", flush=True)
        full_ans.append(chunk)

    ans_text = "".join(full_ans).strip()
    print(f"\n⏱️ ({round(time.time() - t0, 2)}s)\n")

    # Save Short-Term Messages, Extract Long-Term Fact & Trigger Auto-Summarization
    mm.add_message(current_user_id, current_session_id, "user", user_q)
    mm.add_message(current_user_id, current_session_id, "assistant", ans_text)
    extract_and_store_fact(current_user_id, user_q, profile)
    maybe_summarize(current_user_id, current_session_id, profile)