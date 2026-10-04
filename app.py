"""Streamlit UI.   Run:  streamlit run app.py"""
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

import config
import planner

st.set_page_config(page_title=f"{config.ASSISTANT_NAME} | {config.COLLEGE_NAME}",
                   page_icon="🎓", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    padding: 28px 34px; border-radius: 18px; color: white; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(79,70,229,.25);
}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .92;}
.badges {margin-top: 14px;}
.badge {
    display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
    padding: 3px 12px; border-radius: 999px; font-size: .78rem; margin-right: 6px; margin-bottom: 4px;
}
.src {
    display: inline-block; background: rgba(124,58,237,.12); color: #7c3aed; border: 1px solid rgba(124,58,237,.35);
    padding: 2px 10px; border-radius: 999px; font-size: .74rem; margin: 6px 6px 0 0;
}
.card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 16px 18px;
    background: rgba(128,128,128,.06);
}
div[data-testid="stMetric"] {
    background: rgba(124,58,237,.08); border: 1px solid rgba(124,58,237,.25);
    padding: 12px 16px; border-radius: 14px;
}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 10px 18px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ guards
if not os.getenv(config.required_key_name()):
    st.error(f"`{config.required_key_name()}` not found. Add your free key to the `.env` file and restart.")
    st.stop()
if not (config.VECTORSTORE_DIR / "index.faiss").exists():
    st.error("Vector store not built yet. Run `python ingest.py` in the terminal, then refresh.")
    st.stop()

from graph import run_graph          # noqa: E402
from rag import basic_llm_answer     # noqa: E402

# ------------------------------------------------------------------ session
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("history", [])
ss.setdefault("profile", {})
ss.setdefault("plan", None)


def set_pending(q):
    ss["pending"] = q


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"## 🎓 {config.ASSISTANT_NAME}")
    st.caption(f"AI academic assistant for **{config.COLLEGE_NAME}**")
    mode = st.radio(
        "Assistant mode",
        ["RAG Assistant (LangGraph)", "Basic LLM (no documents)"],
        help="Switch to Basic LLM to see how a plain chatbot invents college-specific facts.")


# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <h1>🎓 {config.ASSISTANT_NAME} - AI College Academic Assistant</h1>
  <p>Ask about regulations, exams, courses and internships. Calculate SGPA and attendance. Get a personalized study plan.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ handle a new message (before tabs render)
def handle(prompt):
    ss.messages.append({"role": "user", "content": prompt})
    with st.spinner("NMAMITBot is thinking..."):
        try:
            if mode.startswith("Basic"):
                answer, sources, trace = basic_llm_answer(prompt), [], ["Basic LLM: no retrieval, no tools, no review"]
            else:
                res = run_graph(prompt, ss.history[-8:], ss.profile, ss.plan)
                answer, sources, trace = res["answer"], res.get("sources", []), res.get("trace", [])
                ss.profile = res.get("student_profile", ss.profile)
                ss.plan = res.get("study_plan", ss.plan)
        except Exception as e:                                   # noqa: BLE001
            answer, sources, trace = f"Something went wrong: `{e}`", [], []
    ss.messages.append({"role": "assistant", "content": answer, "sources": sources, "trace": trace})
    ss.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


chat_text = st.chat_input("Ask about attendance, exams, courses, internships, SGPA, study plans...")
prompt = chat_text or ss.pop("pending", None)
if prompt:
    handle(prompt)

# ------------------------------------------------------------------ tabs
tab_chat, tab_plan, tab_arch, tab_eval = st.tabs(
    ["💬 Assistant", "📅 Study Planner", "🧭 Architecture", "📊 Evaluation"])

# ============================== CHAT TAB
with tab_chat:
    if not ss.messages:
        st.markdown("#### 👋 Hi! Try one of these")
        examples = [
            "What is the minimum attendance to write exams?",
            "When is the DBMS exam?",
            "Who teaches Machine Learning?",
            "Summarize the internship guidelines",
            "I attended 33 of 50 classes in OS. How many more to reach 75%?",
            "My grades: DAA A, DBMS O, OS B+ with credits 4,4,3. What's my SGPA?",
            "Create a study plan for DBMS, OS and ML, 3 hours a day, exam on 23 Nov",
            "Who won the 2022 FIFA World Cup?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            cols[i % 2].button(ex, key=f"ex{i}", on_click=set_pending, args=(ex,))

    for m in ss.messages:
        with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
            st.markdown(m["content"])
            if m.get("sources"):
                st.markdown("".join(f'<span class="src">📄 {s}</span>' for s in m["sources"]),
                            unsafe_allow_html=True)
           
# ============================== PLANNER TAB
with tab_plan:
    st.markdown("#### Build a personalized study plan")
    c1, c2 = st.columns(2)
    with c1:
        subjects = st.multiselect("Subjects", config.COURSE_NAMES,
                                  default=["Database Management Systems", "Operating Systems", "Machine Learning"])
        weak = st.multiselect("Weak / harder subjects (get more time)", subjects)
        rest = st.multiselect("Rest days", planner.WEEKDAYS, default=["Sunday"])
    with c2:
        hours = st.slider("Study hours per day", 1.0, 10.0, 3.0, 0.5)
        default_exam = max(date(2026, 11, 23), date.today() + timedelta(days=14))
        exam = st.date_input("Exam date", value=default_exam, min_value=date.today() + timedelta(days=1))
    if st.button("✨ Generate study plan", type="primary"):
        profile = {"subjects": [{"name": s, "difficulty": 3 if s in weak else 2} for s in subjects],
                   "hours_per_day": hours, "exam_date": exam.isoformat(), "rest_weekdays": rest}
        try:
            with st.spinner("Reading the syllabus and building your plan..."):
                ss.plan = planner.build_plan(profile)
            ss.profile = profile
            ss.messages.append({"role": "assistant", "sources": ["Syllabus (B.Tech CSE Sem 5)"], "trace": [],
                                "content": "I've created your study plan from the **Study Planner** tab. "
                                           "Ask me to change it, e.g. *\"make Saturday a rest day\"* or "
                                           "*\"give more time to Machine Learning\"*."})
            ss.history += [{"role": "user", "content": "Generate my study plan"},
                           {"role": "assistant", "content": "Study plan created from the planner tab."}]
        except Exception as e:                                   # noqa: BLE001
            st.error(str(e))

    plan = ss.plan
    if plan:
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Days left", plan["days_left"])
        m2.metric("Study days", plan["study_days"])
        m3.metric("Revision days", plan["revision_days"])
        m4.metric("Total study hours", f"{plan['total_hours']:g}")

        left, right = st.columns([1, 2])
        with left:
            st.markdown("**Hours per subject**")
            st.bar_chart(pd.DataFrame({"Hours": plan["hours_per_subject"]}))
        with right:
            st.markdown("**Day-by-day schedule**")
            rows = []
            for r in plan["rows"]:
                if r["type"] == "rest":
                    text = "😴 Rest day"
                else:
                    tag = "📝 Revision | " if r["type"] == "revision" else ""
                    text = tag + "  •  ".join(f"{s['subject']} ({s['hours']:g}h): {s['focus']}" for s in r["slots"])
                rows.append({"Date": r["date"], "Day": r["weekday"][:3], "Plan": text})
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=380)
        st.download_button("⬇️ Download plan (.md)", planner.plan_to_markdown(plan), file_name="study_plan.md")
        st.info("💡 You can also modify this plan in the Assistant tab, e.g. 'Make Sundays a rest day'.")
    else:
        st.info("Fill in the details above and click **Generate study plan**, or ask the assistant in chat.")

# ============================== ARCHITECTURE TAB
with tab_arch:
    st.markdown("#### LangGraph workflow")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#ede9fe", color="#7c3aed", fontname="Helvetica", fontsize=11];
        edge [color="#6b7280", fontname="Helvetica", fontsize=9];
        start [label="Student\\nquestion", shape=oval, fillcolor="#fce7f3", color="#db2777"];
        qa [label="question_analysis\\n(intent + rewrite)"];
        ret [label="information_retrieval\\n(FAISS + threshold)"];
        tool [label="tool_node\\n(SGPA, attendance,\\ncalendar, calculator)"];
        plan [label="planner_node\\n(create / modify plan)"];
        gen [label="response_generation"];
        rev [label="response_review\\n(grounding check)"];
        fb [label="fallback\\n(not in documents)", fillcolor="#fee2e2", color="#dc2626"];
        end [label="Answer", shape=oval, fillcolor="#dcfce7", color="#16a34a"];
        start -> qa;
        qa -> ret [label="Q&A / course info"];
        qa -> tool [label="calculation"];
        qa -> plan [label="study plan"];
        qa -> gen [label="chit-chat"];
        qa -> fb [label="out of scope"];
        tool -> ret;
        ret -> gen [label="relevant"];
        ret -> fb [label="nothing found"];
        plan -> gen;
        gen -> rev;
        rev -> end [label="grounded"];
        rev -> gen [label="retry once", style=dashed];
        fb -> end;
    }
    """)
    a, b, c = st.columns(3)
    a.markdown('<div class="card"><b>📥 Ingestion</b><br>Load documents → clean → chunk (900 / 150) → '
               'embed with MiniLM → store in FAISS.</div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><b>🔎 Retrieval</b><br>Top-4 chunks with a distance threshold. '
               'Weak matches trigger the fallback node instead of guessing.</div>', unsafe_allow_html=True)
    c.markdown('<div class="card"><b>✅ Review</b><br>An LLM fact-checker verifies every claim against the '
               'retrieved context and forces one strict retry.</div>', unsafe_allow_html=True)

# ============================== EVALUATION TAB
with tab_eval:
    st.markdown("#### Test results and LLM vs RAG comparison")
    t_file, c_file = config.RESULTS_DIR / "test_report.md", config.RESULTS_DIR / "comparison.md"
    if t_file.exists():
        st.markdown(t_file.read_text(encoding="utf-8"))
    else:
        st.info("No test report yet. Run `python tests.py` in the terminal.")
    st.divider()
    if c_file.exists():
        st.markdown(c_file.read_text(encoding="utf-8"))
    else:
        st.info("No comparison yet. Run `python compare.py` in the terminal.")