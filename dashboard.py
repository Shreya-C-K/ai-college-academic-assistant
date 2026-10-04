"""Home dashboard: exam countdown + today's focus."""
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

# Edit these with your real exam dates (YYYY-MM-DD).
EXAMS = [
    {"code": "CS501", "subject": "Design and Analysis of Algorithms", "date": "2026-11-23", "time": "10:00 AM"},
    {"code": "CS502", "subject": "Database Management Systems", "date": "2026-11-26", "time": "10:00 AM"},
    {"code": "CS503", "subject": "Operating Systems", "date": "2026-11-28", "time": "10:00 AM"},
    {"code": "CS504", "subject": "Machine Learning", "date": "2026-12-01", "time": "10:00 AM"},
    {"code": "CS505", "subject": "Computer Networks", "date": "2026-12-04", "time": "10:00 AM"},
    {"code": "CS506", "subject": "Generative AI and LLM Applications", "date": "2026-12-07", "time": "10:00 AM"},
]

CSS = """
<style>
.cd-card {border: 1px solid rgba(128,128,128,.25); border-radius: 16px; padding: 14px 16px;
          background: rgba(128,128,128,.05); margin-bottom: 12px;}
.cd-code {font-size: .78rem; opacity: .65; letter-spacing: .5px;}
.cd-name {font-weight: 600; font-size: .98rem; margin: 2px 0 8px 0; min-height: 2.4em;}
.cd-days {font-size: 2.1rem; font-weight: 700; line-height: 1;}
.cd-unit {font-size: .9rem; font-weight: 500; margin-left: 4px;}
.cd-sub {font-size: .8rem; opacity: .75; margin-top: 6px;}
.cd-red {color: #dc2626;} .cd-orange {color: #d97706;} .cd-green {color: #16a34a;} .cd-grey {color: #6b7280;}
</style>
"""


def _greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def _days_left(iso):
    return (datetime.strptime(iso, "%Y-%m-%d").date() - date.today()).days


def _card(exam):
    d = _days_left(exam["date"])
    nice = datetime.strptime(exam["date"], "%Y-%m-%d").strftime("%a, %d %b %Y")
    if d < 0:
        big, unit, cls = "Done", "", "cd-grey"
    elif d == 0:
        big, unit, cls = "Today!", "", "cd-red"
    else:
        cls = "cd-red" if d <= 7 else "cd-orange" if d <= 14 else "cd-green"
        big, unit = str(d), "day" if d == 1 else "days"
    return (f'<div class="cd-card"><div class="cd-code">{exam["code"]}</div>'
            f'<div class="cd-name">{exam["subject"]}</div>'
            f'<div class="cd-days {cls}">{big}<span class="cd-unit">{unit}</span></div>'
            f'<div class="cd-sub">{nice} · {exam["time"]}</div></div>')


def _today_row(plan):
    iso = date.today().isoformat()
    for row in plan["rows"]:
        if row["date"] == iso:
            return row
    return None


def render_dashboard(plan):
    st.markdown(CSS, unsafe_allow_html=True)
    exams = sorted(EXAMS, key=lambda e: e["date"])
    upcoming = [e for e in exams if _days_left(e["date"]) >= 0]

    st.markdown(f"### {_greeting()}! 👋  Today is {date.today():%A, %d %B %Y}")

    # ---------------- top metrics
    m1, m2, m3 = st.columns(3)
    if upcoming:
        nxt = upcoming[0]
        m1.metric("Next exam in", f"{_days_left(nxt['date'])} days")
        m1.caption(nxt["subject"])
    else:
        m1.metric("Next exam in", "-")
        m1.caption("All exams are over")
    m2.metric("Exams remaining", len(upcoming))
    if plan:
        study_rows = [r for r in plan["rows"] if r["type"] != "rest"]
        done = len([r for r in study_rows if r["date"] < date.today().isoformat()])
        pct = int(100 * done / len(study_rows)) if study_rows else 0
        m3.metric("Study plan progress", f"{pct}%")
        m3.caption(f"{done} of {len(study_rows)} study days completed")
    else:
        m3.metric("Study plan progress", "No plan yet")
        m3.caption("Create one in the Study Planner tab")

    # ---------------- exam countdown
    st.markdown("#### ⏳ Exam countdown")
    for start in range(0, len(exams), 3):
        cols = st.columns(3)
        for col, exam in zip(cols, exams[start:start + 3]):
            col.markdown(_card(exam), unsafe_allow_html=True)

    # ---------------- today's focus
    st.markdown("#### 🎯 Today's focus")
    if not plan:
        st.info("You don't have a study plan yet. Open the **Study Planner** tab (or ask the assistant) "
                "to create one, and your daily tasks will appear here.")
        return

    row = _today_row(plan)
    if row is None:
        st.info(f"Your plan runs from {plan['rows'][0]['date']} to {plan['rows'][-1]['date']}. "
                "Today is outside that range, so please generate a fresh plan.")
    elif row["type"] == "rest":
        st.success("😴 **Rest day.** Recharge today. Your plan resumes tomorrow.")
    else:
        if row["type"] == "revision":
            st.warning("📝 **Revision day.** Revise notes and solve previous-year questions.")
        total, checked = len(row["slots"]), 0
        for slot in row["slots"]:
            label = f"**{slot['subject']}** · {slot['hours']:g} h — {slot['focus']}"
            if st.checkbox(label, key=f"done_{row['date']}_{slot['subject']}"):
                checked += 1
        st.progress(checked / total if total else 0.0)
        if checked == total and total:
            st.balloons()
            st.success("All of today's tasks are done. Great work! 🎉")

    # ---------------- next 7 days
    with st.expander("📆 Next 7 days"):
        wanted = {(date.today() + timedelta(days=i)).isoformat() for i in range(7)}
        rows = []
        for r in plan["rows"]:
            if r["date"] in wanted:
                if r["type"] == "rest":
                    text = "😴 Rest day"
                else:
                    tag = "📝 Revision | " if r["type"] == "revision" else ""
                    text = tag + "  •  ".join(f"{s['subject']} ({s['hours']:g}h)" for s in r["slots"])
                rows.append({"Date": r["date"], "Day": r["weekday"][:3], "Plan": text})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.write("No plan entries for the next 7 days.")