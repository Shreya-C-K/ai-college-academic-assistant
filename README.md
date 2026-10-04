# 🎓 AI-Based College Academic Assistant (NMAM Bot)

A single AI assistant for students of **NMAM  Institute of Technology** (a fictional college with sample data) that can

- answer academic questions and **search college documents** with **RAG**,
- give **course information** (units, credits, faculty, exam dates),
- run **tools** (SGPA calculator, attendance calculator, exam-date calendar, general calculator),
- build and **modify personalized study plans**,
- handle **follow-up questions** and **unknown questions** safely.

Built with **Python · LangChain · LangGraph · FAISS · HuggingFace embeddings · Streamlit** and a free LLM (Groq or Gemini).

---
## 1. Architecture

```
                         ┌──────────────────────── LangGraph workflow ────────────────────────┐
 Student ──► Streamlit ─►│ question_analysis (intent + follow-up rewrite)                      │
                         │   ├─ academic_qa / course_info / document_search / summarize        │
                         │   │        └► information_retrieval (FAISS, relevance threshold)    │
                         │   ├─ calculation ► tool_node ► information_retrieval                │
                         │   ├─ study_plan / modify_plan ► planner_node                        │
                         │   └─ chitchat ► response_generation ; out_of_scope ► fallback       │
                         │ information_retrieval ─(nothing relevant)─► fallback ► END          │
                         │ ... ─► response_generation ─► response_review ─┬─► END              │
                         │                         ▲  (not grounded)      │                    │
                         │                         └──── retry once ◄─────┘                    │
                         └─────────────────────────────────────────────────────────────────────┘
 Offline:  data/*.md|pdf ─► load ─► clean ─► chunk ─► embed (MiniLM) ─► FAISS (vectorstore/)
```

| Requirement in the brief | Where it is implemented |
|---|---|
| Collect documents | `data/` (6 sample documents: syllabus, regulations, exam, internship, FAQs, handbook & calendar) |
| Load, preprocess, chunk, embed, store | `ingest.py` |
| RAG pipeline | `rag.py` + `information_retrieval` node in `graph.py` |
| LangChain: retrieval + prompts + tools + LLM | `prompts.py`, `tools.py`, `llm.py`, LCEL chains in `graph.py` |
| Reusable prompt templates (Q&A, summarization, study planning) | `QA_PROMPT`, `SUMMARY_PROMPT`, `STUDY_PLAN_PROMPT` (+ analysis, review, extraction prompts) |
| Conversational context | chat history is passed to `question_analysis`, which rewrites follow-ups into standalone questions |
| ≥ 1 external tool | `tools.py`: calculator, SGPA, attendance, days-until-exam (calendar) |
| LangGraph workflow with separate nodes | `graph.py`: `question_analysis`, `information_retrieval`, `tool_node`, `planner_node`, `response_generation`, `response_review`, `fallback` |
| Personalized study planner | `planner.py` (subjects, hours/day, exam date, rest days, weak subjects) |
| Modify plan & update workflow | `modify_plan` intent → `planner.modify_profile` → plan rebuilt |
| Unknown questions | similarity threshold + `fallback` node + prompt rule + review node |
| Testing | `tests.py` → `results/test_report.md` |
| Basic LLM vs RAG comparison | `compare.py` → `results/comparison.md`; also a mode switch in the UI |
| Simple UI | `app.py` (Streamlit) |

---
## 2. Setup (VS Code)

```bash
# 1. open the folder in VS Code, then open a terminal (Ctrl + `)
python -m venv venv
venv\Scripts\activate            # Windows        |   source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt

# 2. get a FREE API key (pick one)
#    Groq   (recommended, fast):  https://console.groq.com/keys      -> "Create API Key"
#    Gemini (alternative):        https://aistudio.google.com/apikey -> "Create API key"

# 3. configure
copy .env.example .env           # Windows        |   cp .env.example .env
#    edit .env: set LLM_PROVIDER and paste your key

# 4. build the vector store (first run downloads a ~90 MB embedding model)
python ingest.py

# 5. launch the app
streamlit run app.py
```

Other commands:

```bash
python graph.py              # chat in the terminal and see every node that runs
python debug_retrieval.py    # inspect similarity distances (tune MAX_L2_DISTANCE in .env)
python tests.py              # runs 15 tests -> results/test_report.md
python compare.py            # basic LLM vs RAG -> results/comparison.md / .csv
```

---
## 3. Demo script (≈ 5 minutes)

1. **Direct RAG:** "What is the minimum attendance to write exams?" → 75 %, with source.
2. **Follow-up:** "What if I fall short of it?" → condonation (65-74.9 %, Rs. 500, HoD approval).
3. **Course info:** "Who teaches Machine Learning?" then "What are his office hours?"
4. **Summarize:** "Summarize the internship guidelines."
5. **Tool:** "I attended 33 of 50 classes in OS. How many more do I need for 75%?" → 18.
6. **Tool:** "My grades are DAA A, DBMS O, OS B+, ML A+, CN A with credits 4,4,3,4,3. SGPA?" → 8.50.
7. **Study plan:** "Create a study plan for DBMS, OS and Machine Learning, 3 hours a day, exam on 23 Nov." (or use the sidebar)
8. **Modify:** "Make Sundays a rest day and give more time to Machine Learning."
9. **Unknown:** "Who won the 2022 FIFA World Cup?" and "What is the hostel mess menu?" → polite fallback.
10. **Comparison:** switch the sidebar mode to *Basic LLM* and re-ask question 1 or "What is the condonation fee?"; open the *Workflow trace* expander to show the LangGraph nodes.

---
## 4. Using your own college data
Drop your own `.md`, `.txt` or `.pdf` files into `data/` (delete the samples if you like), update `COURSES` / `COURSE_NAMES` in `config.py`, and re-run `python ingest.py`. Markdown files with `#`/`##`/`###` headings chunk best (each chunk keeps its course/section name).

## 5. Design notes (good for the viva)
- **Why a deterministic planner?** The LLM understands the request and reads unit names from the syllabus (via RAG), but the schedule maths is code, so plans are always valid, reproducible and quick to modify.
- **Why a review node?** It checks the draft against the retrieved context and regenerates once; if it is still unsupported, the safe fallback is returned instead of a hallucination.
- **Unknown handling has three layers:** retrieval distance threshold → `fallback` node, prompt rule ("say you couldn't find it"), and the review node.
- **Free tier tips:** if you hit rate limits, wait a minute or switch `LLM_PROVIDER`. Model names occasionally change; override with `GROQ_MODEL` / `GEMINI_MODEL` in `.env`.

## 6. Troubleshooting
| Problem | Fix |
|---|---|
| `Vector store not found` | run `python ingest.py` |
| `GROQ_API_KEY not found` | check `.env` is in the project root and the key is pasted without quotes |
| Everything answered "couldn't find" | raise `MAX_L2_DISTANCE` (e.g. 1.6) after checking `debug_retrieval.py` |
| Irrelevant questions get answers | lower `MAX_L2_DISTANCE` (e.g. 1.3) |
| Model not found / decommissioned | set a current model name in `.env` |


## Workflow
1. `git checkout main` then `git pull`
2. Create a branch: `git checkout -b feature/short-description`
3. Make your changes and test them (`python tests.py`)
4. `git add .` then `git commit -m "Clear message"`
5. `git push -u origin feature/short-description`
6. Open a Pull Request on GitHub and request a review.

## Rules
- Never push directly to `main`.
- One feature or fix per Pull Request.
- Do not add API keys, `.env` files or the `vectorstore/` folder.
- New documents go in `data/` (Markdown preferred) and need `python ingest.py` to re-index.