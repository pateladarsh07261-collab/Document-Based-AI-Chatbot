import json
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding)


TEST_CASES = [
    
    {"q": "What is a function in Python?", "keywords": ["block", "code", "def", "reusable"], "in_doc": True},
    {"q": "What is the difference between parameters and arguments?", "keywords": ["parameter", "argument", "definition", "pass"], "in_doc": True},
    {"q": "What is a lambda function?", "keywords": ["anonymous", "lambda", "expression", "single"], "in_doc": True},
    {"q": "What is the use of *args in Python?", "keywords": ["positional", "tuple", "variable", "argument"], "in_doc": True},
    {"q": "What is the use of **kwargs in Python?", "keywords": ["keyword", "dictionary", "dict", "argument"], "in_doc": True},
    {"q": "What does the return statement do?", "keywords": ["exit", "return", "value", "caller"], "in_doc": True},
    {"q": "What is recursion in Python?", "keywords": ["itself", "base", "call", "function"], "in_doc": True},
    {"q": "What are default arguments?", "keywords": ["default", "value", "parameter", "optional"], "in_doc": True},
    {"q": "What is a docstring in a function?", "keywords": ["documentation", "string", "__doc__", "triple"], "in_doc": True},
    {"q": "What is the difference between local and global variables?", "keywords": ["local", "global", "scope", "inside"], "in_doc": True},
    {"q": "What is a higher-order function?", "keywords": ["function", "argument", "return", "another"], "in_doc": True},
    {"q": "How does the map() function work?", "keywords": ["iterable", "apply", "function", "element"], "in_doc": True},
    {"q": "What is a decorator in Python?", "keywords": ["wrap", "function", "modify", "behavior"], "in_doc": True},
    {"q": "What is a generator function?", "keywords": ["yield", "iterator", "lazy", "sequence"], "in_doc": True},
    {"q": "What is the difference between return and yield?", "keywords": ["return", "yield", "pause", "state", "terminate"], "in_doc": True},

    
    {"q": "Who won the FIFA World Cup in 2022?", "keywords": [], "in_doc": False},
    {"q": "What is the capital city of France?", "keywords": [], "in_doc": False},
    {"q": "Explain quantum entanglement in physics.", "keywords": [], "in_doc": False},
    {"q": "What is the current stock price of Apple?", "keywords": [], "in_doc": False},
    {"q": "How do you bake a chocolate cake?", "keywords": [], "in_doc": False},
]

rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite this question into a clear standalone search query and append 2 concise related keywords. "
    "Return ONLY the expanded query on one line.\n\nQuestion: {question}\nExpanded Query:"
)

qa_prompt = ChatPromptTemplate.from_template(
    "Answer strictly using only the Context below. "
    "If the answer is not present in the Context, reply exactly: 'Not found in the notes.' "
    "Do NOT use outside knowledge.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def evaluate_pipeline(model_name, rag_mode="advanced"):
    print(f"\n🚀 Running Evaluation -> Model: {model_name} | Mode: {rag_mode.upper()}")
    llm = ChatOllama(model=model_name, temperature=0.0, num_predict=256)
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    qa_chain = qa_prompt | llm | StrOutputParser()

    total_time = 0.0
    retrieval_hits = 0
    answer_relevance_hits = 0
    hallucination_blocked = 0
    detailed_logs = []

    for idx, case in enumerate(TEST_CASES, 1):
        q = case["q"]
        t0 = time.time()

        if rag_mode == "advanced":
            search_query = rewrite_chain.invoke({"question": q}).strip()
            docs = vectorstore.max_marginal_relevance_search(
                search_query, k=4, fetch_k=12, lambda_mult=0.6
            )
        else:
            search_query = q
            results = vectorstore.similarity_search_with_score(search_query, k=4)
            docs = [doc for doc, score in results if score <= 1.5]

        context = "\n\n".join(d.page_content for d in docs) if docs else "No matching notes found."

        if case["in_doc"]:
            context_lower = context.lower()
            if any(kw in context_lower for kw in case["keywords"]):
                retrieval_hits += 1

        answer = qa_chain.invoke({"context": context, "question": q}).strip()
        latency = round(time.time() - t0, 2)
        total_time += latency

        ans_lower = answer.lower()
        refused = "not found in the notes" in ans_lower

        if case["in_doc"]:
            is_relevant = (not refused) and any(kw in ans_lower for kw in case["keywords"])
            if is_relevant:
                answer_relevance_hits += 1
            status = "PASS" if is_relevant else "FAIL"
        else:
            if refused:
                hallucination_blocked += 1
                status = "PASS (Refused Safely)"
            else:
                status = "FAIL (Hallucinated)"

        print(f"  [{idx:02d}/20] {status} ({latency}s) -> {q}")
        detailed_logs.append({
            "question": q,
            "in_doc": case["in_doc"],
            "search_query": search_query,
            "answer": answer,
            "latency_sec": latency,
            "status": status,
        })

    in_doc_total = sum(1 for c in TEST_CASES if c["in_doc"])
    out_doc_total = len(TEST_CASES) - in_doc_total

    summary = {
        "model": model_name,
        "rag_mode": rag_mode,
        "retrieval_accuracy": f"{retrieval_hits}/{in_doc_total} ({round(retrieval_hits / in_doc_total * 100, 1)}%)",
        "answer_relevance": f"{answer_relevance_hits}/{in_doc_total} ({round(answer_relevance_hits / in_doc_total * 100, 1)}%)",
        "faithfulness_no_hallucination": f"{hallucination_blocked}/{out_doc_total} ({round(hallucination_blocked / out_doc_total * 100, 1)}%)",
        "avg_response_time_sec": round(total_time / len(TEST_CASES), 2),
    }
    return summary, detailed_logs


if __name__ == "__main__":
    configs = [
        ("llama3.2:latest", "basic"),
        ("llama3.2:latest", "advanced"),
        ("qwen2.5:3b", "advanced"),
        ("qwen2.5:1.5b", "advanced"),
    ]

    all_summaries = []
    all_details = {}

    for model, mode in configs:
        summary, logs = evaluate_pipeline(model, mode)
        all_summaries.append(summary)
        all_details[f"{model}_{mode}"] = logs

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": all_summaries, "details": all_details}, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# 📊 RAG Evaluation & Model Performance Comparison Report\n",
        "Evaluated across **20 test cases** (15 in-document technical questions + 5 out-of-scope hallucination checks).\n",
        "## Performance Summary Table\n",
        "| Model | RAG Mode | Retrieval Accuracy | Answer Relevance | Faithfulness (No Hallucination) | Avg Latency (sec) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for s in all_summaries:
        md_lines.append(
            f"| `{s['model']}` | **{s['rag_mode'].upper()}** | {s['retrieval_accuracy']} | "
            f"{s['answer_relevance']} | {s['faithfulness_no_hallucination']} | {s['avg_response_time_sec']}s |"
        )

    with open("EVALUATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n✅ Evaluation Complete! Generated 'EVALUATION_REPORT.md' and 'evaluation_results.json'.")