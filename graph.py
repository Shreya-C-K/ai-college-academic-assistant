"""LangGraph workflow.

START -> question_analysis -+-> information_retrieval -+-> response_generation -> response_review -+-> END
                            |        ^                 |                                           |
                            +-> tool_node -------------+  (no info) -> fallback -> END            +-> (retry) response_generation
                            +-> planner_node ----------> response_generation
                            +-> (chitchat) -> response_generation ;  (out_of_scope) -> fallback
"""
import json
import operator
from functools import lru_cache
from typing import Annotated, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

import config
import planner
from llm import get_llm, parse_json
from prompts import (ANALYSIS_PROMPT, CHAT_PROMPT, INTENTS, QA_PROMPT, REVIEW_PROMPT,
                     STUDY_PLAN_PROMPT, SUMMARY_PROMPT)
from rag import format_context, format_history, retrieve, unique_sources
from tools import TOOL_MAP, TOOLS

DOC_INTENTS = {"academic_qa", "course_info", "document_search", "summarize"}
PLAN_INTENTS = {"study_plan", "modify_plan"}

FALLBACK_MSG = (
    "I couldn't find this in the college documents I have access to. "
    "I can help with academic regulations, exams, internships, syllabus and course details, "
    "SGPA/attendance calculations and study plans. For anything else, please contact the "
    "Academic Office (academics@NMAM -tech.example, Admin Block Room 101)."
)


class State(TypedDict, total=False):
    question: str
    chat_history: List[dict]
    standalone_question: str
    intent: str
    context: str
    sources: List[str]
    tool_output: str
    answer: str
    review_passed: bool
    retries: int
    student_profile: dict
    study_plan: dict
    plan_message: str
    trace: Annotated[List[str], operator.add]


def _combined_context(state):
    parts = []
    if state.get("tool_output"):
        parts.append("TOOL RESULTS (exact):\n" + state["tool_output"])
    if state.get("context"):
        parts.append("DOCUMENT EXCERPTS:\n" + state["context"])
    return "\n\n".join(parts) or "(no context)"


# ======================================================================== nodes
def question_analysis(state: State):
    history = state.get("chat_history", [])
    raw = (ANALYSIS_PROMPT | get_llm(0) | StrOutputParser()).invoke({
        "history": format_history(history),
        "has_plan": bool(state.get("study_plan")),
        "question": state["question"]})
    data = parse_json(raw) or {}
    intent = data.get("intent") if data.get("intent") in INTENTS else "academic_qa"
    if intent == "modify_plan" and not state.get("study_plan"):
        intent = "study_plan"
    standalone = data.get("standalone_question") or state["question"]
    return {"intent": intent, "standalone_question": standalone, "retries": 0,
            "trace": [f"question_analysis -> intent='{intent}' | rewritten: {standalone}"]}


def information_retrieval(state: State):
    k = 8 if state["intent"] == "summarize" else config.TOP_K
    kept, raw = retrieve(state["standalone_question"], k)
    best = f"{raw[0][1]:.2f}" if raw else "n/a"
    return {"context": format_context(kept), "sources": unique_sources(kept),
            "trace": [f"information_retrieval -> {len(kept)}/{len(raw)} chunks relevant "
                      f"(best distance {best}, threshold {config.MAX_L2_DISTANCE})"]}


def tool_node(state: State):
    from datetime import date

    llm = get_llm(0).bind_tools(TOOLS)
    msg = llm.invoke([
        SystemMessage(content=f"You are a calculation helper. Today is {date.today():%Y-%m-%d}. "
                              "Call the most suitable tool(s) with exact arguments from the student's message."),
        HumanMessage(content=state["standalone_question"])])
    results, names = [], []
    for call in getattr(msg, "tool_calls", []) or []:
        tool = TOOL_MAP.get(call["name"])
        if not tool:
            continue
        try:
            out = tool.invoke(call["args"])
        except Exception as e:                                   # noqa: BLE001
            out = f"Tool error: {e}"
        results.append(f"{call['name']}({json.dumps(call['args'])}) -> {out}")
        names.append(call["name"])
    return {"tool_output": "\n".join(results),
            "trace": [f"tool_node -> called: {', '.join(names) or 'no tool'}"]}


def planner_node(state: State):
    q, intent = state["question"], state["intent"]
    try:
        if intent == "modify_plan" and state.get("study_plan"):
            profile = planner.modify_profile(q, state["student_profile"])
        else:
            profile = planner.extract_profile(q, state.get("student_profile"))
            missing = planner.missing_fields(profile)
            if missing:
                msg = ("I can build your plan! I still need " + ", ".join(missing) + ". "
                       "For example: *\"Create a study plan for DBMS, Operating Systems and Machine Learning, "
                       "3 hours a day, exam on 23 Nov\"*. You can also fill these in the sidebar.")
                return {"student_profile": profile, "plan_message": msg,
                        "trace": [f"planner_node -> missing info: {missing}"]}
        plan = planner.build_plan(profile)
        return {"student_profile": profile, "study_plan": plan, "plan_message": "",
                "trace": [f"planner_node -> plan {'updated' if intent == 'modify_plan' else 'created'}: "
                          f"{plan['days_left']} days, {plan['total_hours']:g} study hours"]}
    except ValueError as e:
        return {"plan_message": f"I couldn't build the plan: {e}",
                "trace": [f"planner_node -> error: {e}"]}


def response_generation(state: State):
    intent = state["intent"]
    college, assistant = config.COLLEGE_NAME, config.ASSISTANT_NAME
    if intent in PLAN_INTENTS:
        if state.get("plan_message"):
            answer = state["plan_message"]
        else:
            plan = state["study_plan"]
            action = "updated" if intent == "modify_plan" else "created"
            try:
                tips = (STUDY_PLAN_PROMPT | get_llm(0.5) | StrOutputParser()).invoke({
                    "college": college, "action": action, "instruction": state["question"],
                    "summary": planner.plan_summary(plan)})
            except Exception:                                    # noqa: BLE001
                tips = ""
            answer = (f"Done! I've {action} your plan.\n\n" + planner.plan_to_markdown(plan)
                      + (f"\n\n**Coach's tips:** {tips}" if tips else "")
                      + "\n\n*Want changes? Try: \"make Sunday a rest day\" or \"give more time to Machine Learning\".*")
        return {"answer": answer, "trace": ["response_generation -> study plan formatted"]}
    if intent == "chitchat":
        answer = (CHAT_PROMPT | get_llm(0.5) | StrOutputParser()).invoke(
            {"assistant": assistant, "college": college, "question": state["question"]})
        return {"answer": answer, "trace": ["response_generation -> chit-chat reply"]}

    retry = state.get("retries", 0) > 0
    extra = ("IMPORTANT: your previous answer contained claims not found in the context. "
             "Use only facts explicitly written in the CONTEXT.") if retry else ""
    prompt = SUMMARY_PROMPT if intent == "summarize" else QA_PROMPT
    answer = (prompt | get_llm(0.1) | StrOutputParser()).invoke({
        "assistant": assistant, "college": college, "extra": extra,
        "context": _combined_context(state),
        "history": format_history(state.get("chat_history", [])),
        "question": state["standalone_question"]})
    return {"answer": answer,
            "trace": [f"response_generation -> answer drafted{' (strict retry)' if retry else ''}"]}


def response_review(state: State):
    if state["intent"] not in DOC_INTENTS:
        return {"review_passed": True, "trace": ["response_review -> skipped (not document-based)"]}
    raw = (REVIEW_PROMPT | get_llm(0) | StrOutputParser()).invoke({
        "question": state["standalone_question"], "context": _combined_context(state),
        "answer": state["answer"]})
    data = parse_json(raw) or {}
    if data.get("grounded", True):
        return {"review_passed": True, "trace": ["response_review -> PASSED (answer grounded in context)"]}
    retries = state.get("retries", 0)
    reason = data.get("reason", "unsupported claim")
    if retries < config.MAX_REVIEW_RETRIES:
        return {"review_passed": False, "retries": retries + 1,
                "trace": [f"response_review -> FAILED ({reason}); regenerating"]}
    return {"answer": FALLBACK_MSG, "review_passed": True, "sources": [],
            "trace": [f"response_review -> FAILED again ({reason}); safe fallback used"]}


def fallback(state: State):
    return {"answer": FALLBACK_MSG, "sources": [],
            "trace": [f"fallback -> no relevant information (intent='{state.get('intent')}')"]}


# ======================================================================== routing
def route_after_analysis(state: State):
    intent = state["intent"]
    if intent in DOC_INTENTS:
        return "retrieval"
    if intent == "calculation":
        return "tool"
    if intent in PLAN_INTENTS:
        return "planner"
    if intent == "chitchat":
        return "generation"
    return "fallback"


def route_after_retrieval(state: State):
    return "generation" if (state.get("context") or state.get("tool_output")) else "fallback"


def route_after_review(state: State):
    return "end" if state.get("review_passed") else "retry"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(State)
    g.add_node("question_analysis", question_analysis)
    g.add_node("information_retrieval", information_retrieval)
    g.add_node("tool_node", tool_node)
    g.add_node("planner_node", planner_node)
    g.add_node("response_generation", response_generation)
    g.add_node("response_review", response_review)
    g.add_node("fallback", fallback)

    g.add_edge(START, "question_analysis")
    g.add_conditional_edges("question_analysis", route_after_analysis, {
        "retrieval": "information_retrieval", "tool": "tool_node", "planner": "planner_node",
        "generation": "response_generation", "fallback": "fallback"})
    g.add_edge("tool_node", "information_retrieval")
    g.add_conditional_edges("information_retrieval", route_after_retrieval,
                            {"generation": "response_generation", "fallback": "fallback"})
    g.add_edge("planner_node", "response_generation")
    g.add_edge("response_generation", "response_review")
    g.add_conditional_edges("response_review", route_after_review,
                            {"end": END, "retry": "response_generation"})
    g.add_edge("fallback", END)
    return g.compile()


def run_graph(question, history=None, profile=None, plan=None):
    """Run one conversational turn. Returns the final state dict."""
    state = {"question": question, "chat_history": history or [], "student_profile": profile or {},
             "study_plan": plan, "tool_output": "", "context": "", "sources": [],
             "plan_message": "", "trace": []}
    return build_graph().invoke(state)


if __name__ == "__main__":                       # quick CLI chat:  python graph.py
    hist, prof, plan = [], {}, None
    print(f"{config.ASSISTANT_NAME} CLI - type 'exit' to quit")
    while (q := input("\nYou: ").strip()).lower() not in {"exit", "quit"}:
        r = run_graph(q, hist, prof, plan)
        prof, plan = r.get("student_profile", prof), r.get("study_plan", plan)
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]
        print("\n".join("  · " + t for t in r["trace"]))
        print(f"\n{config.ASSISTANT_NAME}: {r['answer']}")
        if r.get("sources"):
            print("Sources:", ", ".join(r["sources"]))
