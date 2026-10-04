"""Compare a plain LLM chatbot with the RAG + LangGraph system.   Run:  python compare.py"""
import csv
import time

import config
from graph import run_graph
from rag import basic_llm_answer

QUESTIONS = [
    "What is the minimum attendance required to write the end-semester exam?",
    "How much is the condonation fee and who approves it?",
    "How many days before the exam are hall tickets released?",
    "Who teaches Machine Learning and what are the office hours?",
    "When is the Database Management Systems end-semester exam?",
    "How long must the mandatory internship be and how is it evaluated?",
    "What is the fee for revaluation of an answer script?",
    "How many books can I borrow from the library and for how many days?",
    "What grade points does an A+ grade carry?",
    "What are the units of CS504 Machine Learning?",
]


def short(text, n=420):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] {q}")
        try:
            basic = basic_llm_answer(q)
        except Exception as e:                                   # noqa: BLE001
            basic = f"ERROR: {e}"
        try:
            res = run_graph(q)
            rag, src = res["answer"], ", ".join(res.get("sources", []))
        except Exception as e:                                   # noqa: BLE001
            rag, src = f"ERROR: {e}", ""
        rows.append({"question": q, "basic_llm": basic, "rag_system": rag, "sources": src})
        time.sleep(1)

    with open(config.RESULTS_DIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Basic LLM chatbot vs RAG-based system", "",
          "| # | Question | Basic LLM (no documents) | RAG system (LangGraph) | Sources |",
          "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['question']} | {short(r['basic_llm']).replace('|', '/')} | "
                  f"{short(r['rag_system']).replace('|', '/')} | {r['sources']} |")
    md += ["", "## Observations", "",
           "- The basic LLM has never seen NMAM  Institute's documents, so it gives generic or invented numbers "
           "(attendance %, fees, dates, faculty names).",
           "- The RAG system retrieves the exact clause and cites its source document.",
           "- Edit this section after reading the table with your own observations."]
    (config.RESULTS_DIR / "comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved results/comparison.md and results/comparison.csv")


if __name__ == "__main__":
    main()
