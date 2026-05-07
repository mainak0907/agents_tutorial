# Roo Code + Local RAG Setup Guide
> Optimise your `.md` instruction files for Roo Code and build a local RAG pipeline from a `.docx` using LangChain — no vector database required.

---

## Table of Contents

1. [Part 1 — Optimised `.md` File for Roo Code](#part-1--optimised-md-file-for-roo-code)
2. [Part 2 — Local RAG from `.docx` + LangChain](#part-2--local-rag-from-docx--langchain-no-vector-db)
3. [Connecting RAG Output to Roo Code](#connecting-rag-output-to-roo-code)
4. [Key Design Decisions](#key-design-decisions)

---

## Part 1 — Optimised `.md` File for Roo Code

Roo Code reads `.md` files as persistent context/instructions (e.g. `CLAUDE.md`, `.roo/rules/*.md`, or custom rule files). The following best practices maximise how well Roo Code parses and acts on your instructions.

---

### Use Clear Role Declarations at the Top

Roo Code parses the first few lines to understand scope. Always declare intent up front:

```md
# Project Instructions
<!-- This file is read by Roo Code as persistent context -->
```

---

### Use H2 Sections for Logical Grouping

Roo Code retrieves context in chunks. Well-separated `##` sections improve retrieval relevance significantly:

```md
## Tech Stack
## Code Style Rules
## File Structure
## Do NOT Rules
```

---

### Be Imperative and Specific

Avoid vague statements — write rules the AI can act on directly:

```md
<!-- ❌ Bad -->
Write good code.

<!-- ✅ Good -->
Always use type hints in Python functions. Never use `print()` for logging; use `logging.info()`.
```

---

### Use Fenced Code Blocks with Language Tags

Roo Code uses these as concrete examples to pattern-match against:

````md
## Example Function Pattern
```python
def process_data(df: pd.DataFrame) -> dict:
    """Always include docstrings."""
    ...
```
````

---

### Use Bullet Lists for Rules

Easier to parse than dense paragraphs:

```md
## Constraints
- Max function length: 40 lines
- All API keys must come from `os.environ`
- Never commit secrets
```

---

### Add a `## Context` Section

Include project-specific terminology and domain knowledge so Roo Code doesn't hallucinate names or concepts:

```md
## Context
This project is a B2B SaaS invoicing tool called "LedgerFlow".
The main data model is `Invoice`, which has states: draft → submitted → approved → paid.
```

---

### Keep It Under ~500 Lines

Very long files get truncated in context windows. Split into multiple focused files if needed:

```
.roo/
  rules-python.md
  rules-api.md
  rules-testing.md
```

---

### Use HTML Comments for Human Notes

`<!-- comments -->` are for humans, not the AI — use them for meta-notes without polluting the instructions:

```md
<!-- TODO: Add more examples for the auth module once it's finalized -->
## Authentication Rules
- Always validate JWT expiry before processing requests
```

---

## Part 2 — Local RAG from `.docx` + LangChain (No Vector DB)

This pipeline uses an **in-memory FAISS index** (included with LangChain) — zero external vector storage, no Docker, no server needed.

---

### Install Dependencies

```bash
pip install langchain langchain-community langchain-core \
            python-docx faiss-cpu sentence-transformers \
            openai  # or use ollama for fully local LLM
```

---

### Full Python Code

```python
# rag_pipeline.py
"""
Local RAG pipeline:
  - Reads a .docx file
  - Chunks it and builds an in-memory FAISS index
  - Reads a Roo Code .md instruction file
  - Answers queries using retrieved context
  - Outputs an updated .md file Roo Code can consume
"""

import os
from pathlib import Path
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ─────────────────────────────────────────────
# 1. Load .docx  →  plain text
# ─────────────────────────────────────────────
def load_docx(path: str) -> str:
    doc = DocxDocument(path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    # Also pull text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    full_text.append(cell.text.strip())
    return "\n\n".join(full_text)


# ─────────────────────────────────────────────
# 2. Load existing .md instruction file
# ─────────────────────────────────────────────
def load_md(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# 3. Chunk  →  in-memory FAISS index
# ─────────────────────────────────────────────
def build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Local embeddings — no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ─────────────────────────────────────────────
# 4. Build RAG chain
# ─────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS, llm):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer.
If the answer isn't in the context, say "Not found in document."

Context:
{context}

Question: {question}
""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────────────────────────────────────
# 5. Enrich the .md file with RAG answers
#    and write output for Roo Code to consume
# ─────────────────────────────────────────────
def enrich_md_with_rag(
    md_content: str,
    chain,
    queries: list[str],
    output_path: str = "CLAUDE.md",
):
    """
    Runs each query through the RAG chain,
    appends the answers to the .md file as a new section,
    and saves it — Roo Code picks this up automatically.
    """
    enriched_sections = []
    for query in queries:
        print(f"[RAG] Querying: {query}")
        answer = chain.invoke(query)
        enriched_sections.append(f"### Q: {query}\n{answer}\n")

    rag_block = "\n## Retrieved Context (Auto-generated)\n\n" + "\n".join(enriched_sections)
    final_md = md_content.rstrip() + "\n\n" + rag_block

    Path(output_path).write_text(final_md, encoding="utf-8")
    print(f"[Done] Updated .md written to: {output_path}")
    return final_md


# ─────────────────────────────────────────────
# 6. Main — wire it all together
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Paths ──
    DOCX_PATH = "your_document.docx"      # ← your .docx
    MD_PATH   = "CLAUDE.md"               # ← your Roo Code instructions .md
    OUT_PATH  = "CLAUDE.md"               # ← overwrite or use a new path

    # ── LLM: choose one ──

    # Option A: OpenAI (needs API key)
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Option B: Ollama (fully local, no key needed)
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3")          # run `ollama pull llama3` first

    # ── Build pipeline ──
    print("[1] Loading .docx...")
    docx_text = load_docx(DOCX_PATH)

    print("[2] Loading .md instructions...")
    md_content = load_md(MD_PATH)

    print("[3] Building in-memory FAISS index...")
    vectorstore = build_vectorstore(docx_text)

    print("[4] Building RAG chain...")
    chain = build_rag_chain(vectorstore, llm)

    # ── Define what to extract from your .docx ──
    # These become enriched context sections in your .md
    queries = [
        "What are the main requirements described in the document?",
        "What technologies or tools are mentioned?",
        "What are the key constraints or rules?",
        "Summarise the project goals in 3 bullet points.",
    ]

    print("[5] Enriching .md with RAG answers...")
    enrich_md_with_rag(md_content, chain, queries, output_path=OUT_PATH)

    # ── Optional: interactive Q&A mode ──
    print("\n[Interactive mode] Type a question or 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("RAG:", chain.invoke(q), "\n")
```

---

## Connecting RAG Output to Roo Code

The pipeline writes an enriched `CLAUDE.md` that Roo Code reads automatically on the next session.

### Configure Roo Code to Watch the File

In your workspace `.roo/config.json`:

```json
{
  "contextFiles": ["CLAUDE.md"],
  "autoReload": true
}
```

### Run the Script Before Your Roo Code Session

```bash
python rag_pipeline.py
# Then open Roo Code — it reads the enriched CLAUDE.md
```

### Or Add It as a VS Code Task

In `.vscode/tasks.json`:

```json
{
  "tasks": [{
    "label": "Refresh RAG Context",
    "type": "shell",
    "command": "python rag_pipeline.py",
    "group": "build",
    "presentation": { "reveal": "silent" }
  }]
}
```

Trigger it with `Ctrl+Shift+B` (or `Cmd+Shift+B`) before starting your coding session.

---

## Key Design Decisions

### Why FAISS In-Memory?

No Docker, no Postgres, no Chroma server required. FAISS runs entirely in RAM and is destroyed when the process ends — perfect for local dev. Zero setup overhead.

### Why `all-MiniLM-L6-v2`?

It's ~80 MB, fast on CPU, and produces strong semantic embeddings. Downloaded once from HuggingFace and cached locally at `~/.cache/huggingface/`.

### Why Enrich the `.md` Instead of Calling RAG Live?

Roo Code reads **static context files**. Pre-computing answers into the `.md` means Roo Code gets the relevant knowledge without any runtime integration or API hooks.

### Optional: Persist the FAISS Index

To skip re-embedding on every run (useful for large `.docx` files):

```python
# Save after building
vectorstore.save_local("faiss_index")

# Load on subsequent runs
vectorstore = FAISS.load_local("faiss_index", embeddings)
```

---

## Quick Reference

| Component | Tool Used | Reason |
|---|---|---|
| `.docx` parsing | `python-docx` | Handles paragraphs + tables |
| Text chunking | `RecursiveCharacterTextSplitter` | Respects natural boundaries |
| Embeddings | `all-MiniLM-L6-v2` | Local, fast, no API key |
| Vector store | `FAISS` (in-memory) | Zero infrastructure |
| LLM (local) | `Ollama` + `llama3` | Fully offline option |
| LLM (cloud) | `ChatOpenAI` | Higher quality, needs key |
| Output | Enriched `CLAUDE.md` | Roo Code reads it as context |

---

*Generated for use with Roo Code extension in VS Code.*
