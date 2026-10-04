"""Load -> clean -> chunk -> embed -> store in FAISS.   Run:  python ingest.py"""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

import config

DOC_TITLES = {
    "01_syllabus_cse_sem5": "Syllabus (B.Tech CSE Sem 5)",
    "02_academic_regulations": "Academic Regulations",
    "03_exam_guidelines": "Examination Guidelines",
    "04_internship_guidelines": "Internship and Placement Guidelines",
    "05_student_faqs": "Student FAQs",
    "06_course_handbook_and_calendar": "Course Handbook and Academic Calendar",
}


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_documents():
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    docs = []
    for path in sorted(config.DATA_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            continue
        title = DOC_TITLES.get(path.stem) or re.sub(r"^\d+_", "", path.stem).replace("_", " ").title()
        for d in loaded:
            d.page_content = clean(d.page_content)
            d.metadata.update({"doc_name": title, "source": path.name})
        docs.extend(loaded)
        print(f"  loaded {path.name:<40} -> {len(loaded)} part(s)")
    return docs


def split_documents(docs):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
    )
    recursive = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for d in docs:
        sections = (header_splitter.split_text(d.page_content)
                    if d.metadata["source"].endswith(".md") else [d])
        for sec in sections:
            meta = {**d.metadata, **sec.metadata}
            for piece in recursive.split_text(sec.page_content):
                crumb = " > ".join(x for x in [meta["doc_name"], meta.get("h2"), meta.get("h3")] if x)
                chunks.append(Document(page_content=f"[{crumb}]\n{piece}", metadata=meta))
    return chunks


def main():
    from langchain_community.vectorstores import FAISS

    from rag import get_embeddings

    print("1/4 Loading documents...")
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No documents found in {config.DATA_DIR}")
    print("2/4 Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"  {len(docs)} documents -> {len(chunks)} chunks "
          f"(avg {sum(len(c.page_content) for c in chunks) // len(chunks)} chars)")
    print("3/4 Embedding chunks (first run downloads the model, ~90 MB)...")
    store = FAISS.from_documents(chunks, get_embeddings())
    print("4/4 Saving vector store...")
    config.VECTORSTORE_DIR.mkdir(exist_ok=True)
    store.save_local(str(config.VECTORSTORE_DIR))
    print(f"Done. Saved to {config.VECTORSTORE_DIR}")


if __name__ == "__main__":
    main()
