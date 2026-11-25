import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

DB_DIR = "chroma_db"

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOllama(model="qwen2.5:3b", temperature=0.1, top_p=0.9, num_predict=250)

vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding)
raw = vectorstore.get()
all_docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"])]

bm25_ret = BM25Retriever.from_documents(all_docs)
bm25_ret.k = 8
vec_ret = vectorstore.as_retriever(search_kwargs={"k": 8})

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the Question into a clear standalone search query for mental wellness notes. "
    "Return ONLY the query on one line.\nQuestion: {question}\nStandalone Query:"
)

qa_prompt = ChatPromptTemplate.from_template(
    "You are an AI Mental Wellness Educational Assistant.\n"
    "Answer strictly using Context in 2-3 concise bullet points. "
    "If missing from Context, reply ONLY: 'Not found in the knowledge base.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# 20 Test Cases: (Question, Expected Source File, Expected Keyword in Answer)
TEST_CASES = [
    ("What is stress?", "stress_management.txt", "response"),
    ("Difference between acute and chronic stress?", "stress_management.txt", "chronic"),
    ("How does chronic stress affect health?", "stress_management.txt", "sleep"),
    ("Give tips to manage daily stress.", "stress_management.txt", "movement"),
    ("How does physical exercise help with stress?", "stress_management.txt", "walking"),
    ("What is anxiety?", "anxiety_awareness.txt", "worry"),
    ("What are common physical symptoms of anxiety?", "anxiety_awareness.txt", "heart"),
    ("Explain the 5-4-3-2-1 grounding technique.", "anxiety_awareness.txt", "5"),
    ("How to calm down during sudden panic?", "anxiety_awareness.txt", "4-7-8"),
    ("When should someone seek professional help for anxiety?", "anxiety_awareness.txt", "professional"),
    ("What is sleep hygiene?", "mindfulness_sleep.txt", "sleep hygiene"),
    ("How many hours of sleep do adults need?", "mindfulness_sleep.txt", "7"),
    ("Why should screens be avoided before bedtime?", "mindfulness_sleep.txt", "screen"),
    ("What is the ideal bedroom environment for sleep?", "mindfulness_sleep.txt", "dark"),
    ("How does caffeine affect sleep quality?", "mindfulness_sleep.txt", "caffeine"),
    ("What is mindfulness?", "mindfulness_sleep.txt", "present"),
    ("Explain box breathing technique.", "mindfulness_sleep.txt", "breath"),
    ("What is emotional wellbeing?", "emotional_wellbeing.txt", "emotional"),
    ("How can journaling improve emotional well-being?", "emotional_wellbeing.txt", "journal"),
    ("What medicine should I take for severe depression?", "NONE", "Not found in the knowledge base"),
]


def retrieve_docs(mode, question):
    if mode == "Basic RAG":
        return vectorstore.similarity_search(question, k=3)

    if mode == "Advanced RAG":
        sq = (rewrite_prompt | llm | StrOutputParser()).invoke({"question": question}).strip()
        return vectorstore.similarity_search(sq, k=3)

    # Hybrid RRF (used in both Hybrid Search & Hybrid + Rerank)
    bm25_docs = bm25_ret.invoke(question)
    vec_docs = vec_ret.invoke(question)
    scores_map, doc_map = {}, {}
    for rank, d in enumerate(bm25_docs):
        doc_map[d.page_content] = d
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.4 / (rank + 60)
    for rank, d in enumerate(vec_docs):
        doc_map[d.page_content] = d
        scores_map[d.page_content] = scores_map.get(d.page_content, 0.0) + 0.6 / (rank + 60)

    candidates = [doc_map[k] for k, _ in sorted(scores_map.items(), key=lambda x: x[1], reverse=True)][:8]

    if mode == "Hybrid Search":
        return candidates[:3]

    # Hybrid + Rerank (Day 3)
    pairs = [[question, d.page_content] for d in candidates]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:3]]


modes = ["Basic RAG", "Advanced RAG", "Hybrid Search", "Hybrid + Rerank"]
summary = {}

for mode in modes:
    print(f"\n🚀 Running Benchmark: {mode} (20 Queries)...")
    rel_hits, ans_hits, total_time = 0, 0, 0.0
    failed_queries = []

    for q, exp_src, exp_kw in TEST_CASES:
        t0 = time.time()
        docs = retrieve_docs(mode, q)
        ctx = "\n\n".join(d.page_content for d in docs)
        ans = (qa_prompt | llm | StrOutputParser()).invoke({"context": ctx, "question": q}).strip()
        total_time += (time.time() - t0)

        retrieved_sources = [d.metadata.get("source") for d in docs]
        src_ok = True if exp_src == "NONE" else (exp_src in retrieved_sources)
        ans_ok = exp_kw.lower() in ans.lower()

        if src_ok:
            rel_hits += 1
        if ans_ok:
            ans_hits += 1
        if not (src_ok and ans_ok):
            failed_queries.append((q, retrieved_sources, ans[:60]))

    summary[mode] = {
        "Retrieval Relevance": f"{(rel_hits / len(TEST_CASES)) * 100:.1f}% ({rel_hits}/20)",
        "Answer Accuracy": f"{(ans_hits / len(TEST_CASES)) * 100:.1f}% ({ans_hits}/20)",
        "Avg Time/Query": f"{total_time / len(TEST_CASES):.2f}s",
        "Failures": failed_queries,
    }

print("\n" + "=" * 75)
print("📊 FINAL 4-WAY RAG EVALUATION REPORT (20 TEST CASES)")
print("=" * 75)
print(f"{'Pipeline Mode':<20} | {'Retrieval Relevance':<22} | {'Answer Accuracy':<20} | {'Avg Time':<10}")
print("-" * 75)
for mode, stats in summary.items():
    print(f"{mode:<20} | {stats['Retrieval Relevance']:<22} | {stats['Answer Accuracy']:<20} | {stats['Avg Time/Query']:<10}")

print("\n🔍 Failure / Edge-Case Observations:")
for mode, stats in summary.items():
    print(f"\n[{mode}] Failed Cases ({len(stats['Failures'])}):")
    for fq, srcs, snippet in stats["Failures"][:3]:
        print(f"  - Q: '{fq}' | Got Sources: {srcs} | Ans: '{snippet}...'")