"""Automated test-suite: direct, follow-up, RAG-based, unknown and multi-step requests.
Run:  python tests.py      -> prints a summary and writes results/test_report.md
"""
import time

import config
from graph import run_graph

UNKNOWN = ["couldn't find", "could not find", "can only help", "not available"]

# Each test: category, name, turns = [(message, [any-of keywords expected in the answer])]
TESTS = [
    ("Direct", "Attendance rule", [("What is the minimum attendance required to write exams?", ["75"])]),
    ("Direct", "Grade points", [("What grade points does an A+ carry?", ["9"])]),
    ("Direct", "Hall ticket", [("When are hall tickets released?", ["10 days"])]),
    ("Follow-up", "Faculty then office hours",
     [("Who teaches Machine Learning?", ["rohan mehta"]),
      ("What are his office hours?", ["tue", "thu", "3:00"])]),
    ("Follow-up", "Attendance then condonation",
     [("What is the minimum attendance requirement?", ["75"]),
      ("What if I fall short of it?", ["condon"])]),
    ("RAG-based", "Exam date", [("When is the DBMS end-semester exam?", ["26 nov"])]),
    ("RAG-based", "Internship length", [("How long should the mandatory internship be?", ["8 weeks", "8-week"])]),
    ("RAG-based", "Calculator policy", [("Can I use a calculator in the exam hall?", ["casio", "non-programmable"])]),
    ("RAG-based", "Syllabus units", [("What topics are covered in Unit 3 of Operating Systems?", ["deadlock"])]),
    ("Unknown", "General knowledge", [("Who won the FIFA World Cup in 2022?", UNKNOWN)]),
    ("Unknown", "Missing college info", [("What is the Sunday hostel mess menu?", UNKNOWN)]),
    ("Unknown", "Other programme", [("What is the fee structure of the MBA programme?", UNKNOWN)]),
    ("Tool / calc", "SGPA",
     [("My grades are DAA A, DBMS O, OS B+, ML A+, CN A with credits 4,4,3,4,3. What is my SGPA?", ["8.5"])]),
    ("Tool / calc", "Attendance classes needed",
     [("I attended 33 of 50 classes in Operating Systems. How many more must I attend to reach 75%?", ["18"])]),
    ("Multi-step", "Create then modify plan",
     [("Create a study plan for DBMS, Operating Systems and Machine Learning, 3 hours a day, exam on 23 Nov 2026",
       ["week 1"]),
      ("Make Sundays a rest day and give more time to Machine Learning", ["rest day"])]),
]


def run_test(turns):
    history, profile, plan, answer, trace = [], {}, None, "", []
    for msg, _ in turns:
        res = run_graph(msg, history, profile, plan)
        profile, plan = res.get("student_profile", profile), res.get("study_plan", plan)
        answer, trace = res["answer"], res.get("trace", [])
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": answer}]
        time.sleep(0.5)
    return answer, trace, profile


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    rows, passed = [], 0
    for cat, name, turns in TESTS:
        try:
            answer, trace, profile = run_test(turns)
            expected = turns[-1][1]
            ok = any(k.lower() in answer.lower() for k in expected)
            if name == "Create then modify plan":
                ok = ok and "Sunday" in (profile.get("rest_weekdays") or [])
        except Exception as e:                                   # noqa: BLE001
            answer, trace, ok = f"ERROR: {e}", [], False
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {cat:<11} {name}")
        rows.append((cat, name, turns[-1][0], " ".join(answer.split())[:350], trace, ok))

    md = [f"# Test Report - {passed}/{len(TESTS)} passed", "",
          "| Category | Test | Last question | Answer (truncated) | Result |", "|---|---|---|---|---|"]
    for cat, name, q, a, _, ok in rows:
        md.append(f"| {cat} | {name} | {q.replace('|', '/')} | {a.replace('|', '/')} | {'PASS' if ok else 'FAIL'} |")
    (config.RESULTS_DIR / "test_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{passed}/{len(TESTS)} passed. Report: results/test_report.md")


if __name__ == "__main__":
    main()
