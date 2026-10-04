"""Shows retrieval distances so you can tune MAX_L2_DISTANCE.   Run:  python debug_retrieval.py"""
from rag import retrieve

QUERIES = [
    "minimum attendance for exams",            # should be relevant (low distance)
    "when is the DBMS exam",
    "what is the hostel mess menu",            # unknown
    "who won the football world cup",          # unknown
]
for q in QUERIES:
    _, raw = retrieve(q, k=3)
    print(f"\nQ: {q}")
    for d, s in raw:
        print(f"   distance={s:.3f}  [{d.metadata['doc_name']}]  {d.page_content[:70]!r}")
