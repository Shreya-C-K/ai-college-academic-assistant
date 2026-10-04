"""Personalized study planner.

The schedule itself is built deterministically (reliable, explainable); the LLM is only used to
(1) understand the student's message, (2) read unit titles from the syllabus through RAG.
"""
import json
import math
from datetime import date, datetime, timedelta

import config

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0}          # difficulty -> share of study time
_topic_cache = {}


# ------------------------------------------------------------------ helpers
def _norm_day(s):
    s = str(s).strip().lower()
    if len(s) < 3:
        return None
    for d in WEEKDAYS:
        if d.lower().startswith(s[:3]):
            return d
    return None


def _norm_days(items):
    return sorted({d for d in (_norm_day(x) for x in (items or [])) if d}, key=WEEKDAYS.index)


def _norm_subject(name):
    """Map free text to an official course name when possible."""
    n = str(name).strip().lower()
    for official in config.COURSE_NAMES:
        if n == official.lower() or n in official.lower() or official.lower() in n:
            return official
    return str(name).strip()


def _valid_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------ syllabus topics via RAG
def get_topics(subject):
    if subject in _topic_cache:
        return _topic_cache[subject]
    topics = []
    try:
        from langchain_core.output_parsers import StrOutputParser

        from llm import get_llm, parse_json
        from prompts import TOPICS_PROMPT
        from rag import format_context, retrieve

        kept, _ = retrieve(f"{subject} syllabus units topics", k=6)
        if kept:
            raw = (TOPICS_PROMPT | get_llm(0) | StrOutputParser()).invoke(
                {"subject": subject, "context": format_context(kept)})
            data = parse_json(raw)
            if isinstance(data, list):
                topics = [str(t) for t in data if str(t).strip()]
    except Exception:
        topics = []
    if not topics:
        topics = [f"Unit {i}" for i in range(1, 6)]
    _topic_cache[subject] = topics
    return topics


# ------------------------------------------------------------------ profile handling
def extract_profile(question, base=None):
    """Understand a natural-language request and merge it into the existing profile."""
    from langchain_core.output_parsers import StrOutputParser

    from llm import get_llm, parse_json
    from prompts import PROFILE_PROMPT

    base = dict(base or {})
    raw = (PROFILE_PROMPT | get_llm(0) | StrOutputParser()).invoke({
        "today": date.today().isoformat(), "courses": "\n".join(config.COURSES),
        "current": json.dumps(base), "question": question})
    data = parse_json(raw) or {}
    profile = dict(base)
    if data.get("subjects"):
        profile["subjects"] = [
            {"name": _norm_subject(s.get("name", "")), "difficulty": int(s.get("difficulty") or 2)}
            for s in data["subjects"] if s.get("name")]
    if data.get("hours_per_day"):
        profile["hours_per_day"] = float(data["hours_per_day"])
    if _valid_date(data.get("exam_date")):
        profile["exam_date"] = data["exam_date"]
    if data.get("rest_weekdays") is not None:
        profile["rest_weekdays"] = _norm_days(data["rest_weekdays"])
    profile.setdefault("rest_weekdays", [])
    return profile


def missing_fields(profile):
    missing = []
    if not profile.get("subjects"):
        missing.append("which subjects you want to study")
    if not profile.get("hours_per_day"):
        missing.append("how many hours per day you can study")
    if not profile.get("exam_date"):
        missing.append("your exam date")
    return missing


def modify_profile(instruction, profile):
    """Turn 'make Sunday a rest day' etc. into changes on the stored profile."""
    from langchain_core.output_parsers import StrOutputParser

    from llm import get_llm, parse_json
    from prompts import MODIFY_PROMPT

    raw = (MODIFY_PROMPT | get_llm(0) | StrOutputParser()).invoke({
        "today": date.today().isoformat(), "courses": "\n".join(config.COURSES),
        "profile": json.dumps(profile), "instruction": instruction})
    return apply_changes(profile, parse_json(raw) or {})


def apply_changes(profile, ch):
    p = json.loads(json.dumps(profile))                      # deep copy
    if ch.get("hours_per_day"):
        p["hours_per_day"] = float(ch["hours_per_day"])
    if _valid_date(ch.get("exam_date")):
        p["exam_date"] = ch["exam_date"]
    rest = set(p.get("rest_weekdays") or [])
    rest |= set(_norm_days(ch.get("rest_weekdays_add")))
    rest -= set(_norm_days(ch.get("rest_weekdays_remove")))
    p["rest_weekdays"] = sorted(rest, key=WEEKDAYS.index)
    subjects = p.get("subjects", [])
    for name in ch.get("remove_subjects") or []:
        subjects = [s for s in subjects if s["name"] != _norm_subject(name)]
    for s in ch.get("add_subjects") or []:
        nm = _norm_subject(s.get("name", ""))
        if nm and all(x["name"] != nm for x in subjects):
            subjects.append({"name": nm, "difficulty": int(s.get("difficulty") or 2)})
    for name, diff in (ch.get("difficulty_changes") or {}).items():
        for s in subjects:
            if s["name"] == _norm_subject(name):
                s["difficulty"] = int(diff)
    p["subjects"] = subjects
    return p


# ------------------------------------------------------------------ the scheduler
def build_plan(profile, topics_fn=None, today=None):
    topics_fn = topics_fn or get_topics
    today = today or date.today()
    exam = _valid_date(profile.get("exam_date"))
    if not exam or exam <= today:
        raise ValueError("The exam date must be a future date.")
    subjects = profile.get("subjects") or []
    if not subjects:
        raise ValueError("Please tell me which subjects to include.")
    hours = float(profile.get("hours_per_day") or 0)
    if hours <= 0:
        raise ValueError("Please tell me how many hours per day you can study.")

    rest = set(profile.get("rest_weekdays") or [])
    all_days = [today + timedelta(days=i) for i in range((exam - today).days)]
    study_days = [d for d in all_days if d.strftime("%A") not in rest]
    if not study_days:
        raise ValueError("Every day before the exam is marked as a rest day.")

    n_rev = 2 if len(study_days) >= 8 else 1 if len(study_days) >= 4 else 0
    learn = study_days[: len(study_days) - n_rev]
    revise = study_days[len(study_days) - n_rev:]
    bpd = max(1, round(hours * 2))                               # half-hour blocks per day

    names = [s["name"] for s in subjects]
    weights = {s["name"]: WEIGHT.get(int(s.get("difficulty") or 2), 1.5) for s in subjects}
    wsum = sum(weights.values())
    topics = {n: topics_fn(n) for n in names}

    rows = {}
    # Phase 1: learning, weighted by difficulty, syllabus units spread proportionally
    target = {n: len(learn) * bpd * weights[n] / wsum for n in names}
    credit, cum = {n: 0.0 for n in names}, {n: 0 for n in names}
    for d in learn:
        alloc = {n: 0 for n in names}
        for n in names:                                  # each day every subject earns its weighted share
            credit[n] += bpd * weights[n] / wsum
        for _ in range(bpd):
            n = max(names, key=lambda x: credit[x])
            alloc[n] += 1
            credit[n] -= 1
        slots = []
        for n in names:
            if not alloc[n]:
                continue
            t, tot = topics[n], max(target[n], 1e-9)
            s = min(len(t) - 1, int(cum[n] / tot * len(t)))
            e = min(len(t) - 1, max(s, math.ceil((cum[n] + alloc[n]) / tot * len(t)) - 1))
            cum[n] += alloc[n]
            slots.append({"subject": n, "hours": alloc[n] / 2, "focus": "; ".join(t[s:e + 1])})
        rows[d] = {"type": "study", "slots": slots}

    # Phase 2: revision days
    rcredit = {n: 0.0 for n in names}
    for d in revise:
        alloc = {n: 0 for n in names}
        for n in names:
            rcredit[n] += bpd * weights[n] / wsum
        for _ in range(bpd):
            n = max(names, key=lambda x: rcredit[x])
            alloc[n] += 1
            rcredit[n] -= 1
        rows[d] = {"type": "revision", "slots": [
            {"subject": n, "hours": alloc[n] / 2, "focus": "Revise notes, formulas and previous-year questions"}
            for n in names if alloc[n]]}

    out_rows, per_subject = [], {n: 0.0 for n in names}
    for d in all_days:
        r = rows.get(d, {"type": "rest", "slots": []})
        for s in r["slots"]:
            per_subject[s["subject"]] += s["hours"]
        out_rows.append({"date": d.isoformat(), "weekday": d.strftime("%A"), **r})
    return {
        "profile": profile, "exam_date": exam.isoformat(), "days_left": len(all_days),
        "study_days": len(study_days), "revision_days": len(revise),
        "rows": out_rows, "hours_per_subject": per_subject,
        "total_hours": sum(per_subject.values()),
    }


# ------------------------------------------------------------------ presentation
def plan_summary(plan):
    p = plan["profile"]
    per = ", ".join(f"{k}: {v:g}h" for k, v in plan["hours_per_subject"].items())
    return (f"Exam on {plan['exam_date']}; {plan['days_left']} days left, {plan['study_days']} study days "
            f"({plan['revision_days']} revision days); {p['hours_per_day']:g} h/day; "
            f"rest days: {', '.join(p.get('rest_weekdays') or []) or 'none'}; "
            f"total {plan['total_hours']:g}h => {per}")


def plan_to_markdown(plan):
    p = plan["profile"]
    lines = [
        f"### Your personalized study plan (exam: {datetime.fromisoformat(plan['exam_date']):%d %b %Y})",
        f"- **Days left:** {plan['days_left']} | **Study days:** {plan['study_days']} "
        f"(last {plan['revision_days']} are revision) | **Hours/day:** {p['hours_per_day']:g}",
        f"- **Rest days:** {', '.join(p.get('rest_weekdays') or []) or 'none'}",
        "- **Time per subject:** " + ", ".join(
            f"{k} **{v:g}h**" for k, v in plan["hours_per_subject"].items()),
        "",
    ]
    rows = plan["rows"]
    for w in range(0, len(rows), 7):
        chunk = rows[w:w + 7]
        lines += [f"**Week {w // 7 + 1}**", "", "| Date | Day | Plan |", "|---|---|---|"]
        for r in chunk:
            d = datetime.fromisoformat(r["date"])
            if r["type"] == "rest":
                cell = "Rest day"
            else:
                tag = "Revision: " if r["type"] == "revision" else ""
                cell = tag + " / ".join(
                    f"**{s['subject']}** ({s['hours']:g}h): {s['focus']}" for s in r["slots"])
            lines.append(f"| {d:%d %b} | {r['weekday'][:3]} | {cell} |")
        lines.append("")
    return "\n".join(lines)
