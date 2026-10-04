"""Quiz Me: AI-generated practice quizzes with weak-topic tracking."""
import random
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
import planner
from llm import get_llm, parse_json
from rag import format_context, retrieve

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an exam-setter at {college}.
Write exactly {n} multiple-choice questions for the course "{subject}". Scope: {unit}. Difficulty: {difficulty}.

Use the SYLLABUS EXCERPT only to decide WHICH topics to cover. Use your own subject knowledge to write accurate,
conceptual questions (not questions about the syllabus document itself, unit numbers or textbooks).

Rules:
- Exactly 4 options per question and exactly one correct option.
- Wrong options must be plausible. Never use "all of the above" or "none of the above".
- "answer" is the index (0 to 3) of the correct option.
- "topic" is a short topic name (max 4 words) taken from the scope.
- "explanation" is 1-2 sentences explaining why the answer is correct.

Return ONLY a JSON list, with no extra text:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0, "topic": "...", "explanation": "..."}}]"""),
    ("human", "SYLLABUS EXCERPT:\n{excerpt}"),
])


# ------------------------------------------------------------------ generation
def _to_index(ans):
    if isinstance(ans, bool):
        return None
    if isinstance(ans, int):
        return ans
    s = str(ans).strip().upper()
    if s in ("A", "B", "C", "D"):
        return "ABCD".index(s)
    return int(s) if s.isdigit() else None


def _clean(data, n):
    """Validate the LLM output, shuffle options so the answer isn't always A."""
    if isinstance(data, dict):
        data = data.get("questions")
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = item.get("options")
        idx = _to_index(item.get("answer"))
        if not question or not isinstance(options, list) or len(options) != 4:
            continue
        if idx is None or not 0 <= idx <= 3:
            continue
        options = [re.sub(r"^[A-Da-d][\).:]\s+", "", str(o).strip()) for o in options]
        if any(not o for o in options) or len(set(options)) < 4:
            continue
        correct = options[idx]
        random.shuffle(options)
        out.append({
            "question": question,
            "options": options,
            "answer": options.index(correct),
            "topic": str(item.get("topic") or "General").strip() or "General",
            "explanation": str(item.get("explanation") or "").strip(),
        })
    return out[:n]


def generate_quiz(subject, unit, n=5, difficulty="Medium"):
    query = f"{subject} syllabus units topics" if unit == "All units" else f"{subject} {unit}"
    kept, raw = retrieve(query, k=6)
    docs = kept or raw[:3]
    if not docs:
        raise ValueError("I couldn't find this subject in the syllabus documents.")
    chain = QUIZ_PROMPT | get_llm(0.6) | StrOutputParser()
    for _ in range(2):                                           # one automatic retry
        reply = chain.invoke({
            "college": config.COLLEGE_NAME, "n": n, "subject": subject,
            "unit": unit, "difficulty": difficulty, "excerpt": format_context(docs)})
        questions = _clean(parse_json(reply), n)
        if len(questions) >= min(3, n):
            return questions
    raise ValueError("The AI could not produce a valid quiz this time. Please click Generate again.")


# ------------------------------------------------------------------ helpers
def _labels(q):
    return [f"{'ABCD'[j]}. {opt}" for j, opt in enumerate(q["options"])]


def _evaluate(quiz):
    correct, per_topic = 0, {}
    for i, q in enumerate(quiz["questions"]):
        ok = quiz["answers"].get(i) == q["answer"]
        correct += ok
        stat = per_topic.setdefault(q["topic"], [0, 0])
        stat[0] += ok
        stat[1] += 1
    return correct, per_topic


def _record(quiz):
    ss = st.session_state
    correct, per_topic = _evaluate(quiz)
    total = len(quiz["questions"])
    ss.quiz_history.append({
        "Time": datetime.now().strftime("%d %b %H:%M"), "Subject": quiz["subject"],
        "Scope": quiz["unit"], "Score": f"{correct}/{total}", "Percent": round(100 * correct / total)})
    for topic, (c, t) in per_topic.items():
        stat = ss.topic_stats.setdefault(f"{quiz['subject']} - {topic}", [0, 0])
        stat[0] += c
        stat[1] += t


# ------------------------------------------------------------------ UI
def render_quiz_tab():
    ss = st.session_state
    ss.setdefault("quiz", None)
    ss.setdefault("quiz_id", 0)
    ss.setdefault("quiz_history", [])
    ss.setdefault("topic_stats", {})

    st.markdown("#### 📝 Quiz Me: test yourself and find your weak spots")
    c1, c2, c3, c4 = st.columns([3, 3, 1.5, 1.5])
    subject = c1.selectbox("Subject", config.COURSE_NAMES, key="quiz_subject")
    with c2:
        with st.spinner("Loading units..."):
            units = ["All units"] + planner.get_topics(subject)
        unit = st.selectbox("Unit / topic", units, key=f"quiz_unit_{subject}")
    n_questions = c3.selectbox("Questions", [5, 8, 10], key="quiz_n")
    difficulty = c4.selectbox("Level", ["Easy", "Medium", "Hard"], index=1, key="quiz_level")

    if st.button("🎲 Generate quiz", type="primary"):
        try:
            with st.spinner("Writing your quiz..."):
                questions = generate_quiz(subject, unit, n_questions, difficulty)
            ss.quiz_id += 1
            ss.quiz = {"subject": subject, "unit": unit, "difficulty": difficulty,
                       "questions": questions, "answers": {}, "submitted": False}
        except Exception as e:                                   # noqa: BLE001
            ss.quiz = None
            st.error(str(e))

    quiz = ss.quiz
    if quiz:
        st.divider()
        qs = quiz["questions"]
        if not quiz["submitted"]:
            with st.form(f"quiz_form_{ss.quiz_id}"):
                st.markdown(f"**{quiz['subject']}** · {quiz['unit']} · {quiz['difficulty']}")
                for i, q in enumerate(qs):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.radio("Choose one", _labels(q), index=None,
                             key=f"q_{ss.quiz_id}_{i}", label_visibility="collapsed")
                submitted = st.form_submit_button("✅ Submit answers")
            if submitted:
                for i, q in enumerate(qs):
                    choice = ss.get(f"q_{ss.quiz_id}_{i}")
                    labels = _labels(q)
                    quiz["answers"][i] = labels.index(choice) if choice in labels else None
                quiz["submitted"] = True
                _record(quiz)
                st.rerun()
        else:
            correct, per_topic = _evaluate(quiz)
            total = len(qs)
            pct = round(100 * correct / total)
            r1, r2 = st.columns([1, 3])
            r1.metric("Your score", f"{correct}/{total}", f"{pct}%", delta_color="off")
            with r2:
                st.progress(pct / 100)
                if pct >= 80:
                    st.success("Excellent! You're well prepared for this part. 🌟")
                elif pct >= 60:
                    st.info("Good effort. A little more revision will make it solid. 👍")
                else:
                    st.warning("This area needs more practice. Review the topics below. 📚")

            weak = [t for t, (c, tot) in per_topic.items() if c < tot]
            if weak:
                st.markdown("**Topics to revise:** " + ", ".join(f"`{t}`" for t in weak))

            for i, q in enumerate(qs):
                chosen = quiz["answers"].get(i)
                ok = chosen == q["answer"]
                with st.expander(f"{'✅' if ok else '❌'} Q{i + 1}. {q['question']}", expanded=not ok):
                    st.markdown("**Your answer:** " + (_labels(q)[chosen] if chosen is not None else "_not answered_"))
                    st.markdown("**Correct answer:** " + _labels(q)[q["answer"]])
                    if q["explanation"]:
                        st.markdown(f"💡 {q['explanation']}")

            # connect the quiz to the study planner
            profile, plan = ss.get("profile") or {}, ss.get("plan")
            plan_subjects = [s["name"] for s in profile.get("subjects", [])]
            if pct < 70 and plan and quiz["subject"] in plan_subjects:
                if st.button(f"📈 Give {quiz['subject']} more time in my study plan"):
                    try:
                        new_profile = planner.apply_changes(
                            profile, {"difficulty_changes": {quiz["subject"]: 3}})
                        ss.plan = planner.build_plan(new_profile)
                        ss.profile = new_profile
                        st.success("Done! Your study plan now gives this subject extra time. "
                                   "Check the Study Planner tab.")
                    except ValueError as e:
                        st.error(str(e))
            elif pct < 70 and not plan:
                st.caption("💡 Create a study plan and I can automatically add extra time for weak subjects.")

            b1, b2 = st.columns(2)
            if b1.button("🔁 Retry the same questions"):
                quiz["submitted"], quiz["answers"] = False, {}
                ss.quiz_id += 1
                st.rerun()
            if b2.button("🆕 New quiz"):
                ss.quiz = None
                st.rerun()

    # ---------------- progress tracker
    if ss.quiz_history:
        st.divider()
        st.markdown("#### 📊 Your quiz progress")
        hist = pd.DataFrame(ss.quiz_history)
        p1, p2 = st.columns([2, 1])
        with p1:
            st.dataframe(hist, hide_index=True)
        with p2:
            st.bar_chart(hist.set_index(hist.index + 1)["Percent"])
        stats = [(topic, c / t, t) for topic, (c, t) in ss.topic_stats.items() if t > 0]
        if stats:
            stats.sort(key=lambda x: x[1])
            st.markdown("**Your weakest topics so far**")
            st.dataframe(pd.DataFrame(
                [{"Topic": t, "Accuracy": f"{round(100 * a)}%", "Questions attempted": n}
                 for t, a, n in stats[:5]]), hide_index=True)