
# 📚 RAG for Books & PDFs

A complete **Retrieval-Augmented Generation (RAG) system** that allows AI to answer questions using books, research papers, manuals, or any PDF documents. This system extracts text, chunks it, stores semantic embeddings, retrieves relevant passages, and generates LLM-based answers with citations.

---
### This is the break down of all the process which needs to be done but in code i have created a single complete version of it

## Table of Contents

1. [Features](#features)  
2. [Tech Stack](#tech-stack)  
3. [Setup & Installation](#setup--installation)  
4. [PDF Ingestion](#pdf-ingestion)  
5. [Text Extraction (with OCR)](#text-extraction-with-ocr)  
6. [Chunking & Metadata](#chunking--metadata)  
7. [Embeddings & Vector Store](#embeddings--vector-store)  
8. [Retrieval & Optional Reranking](#retrieval--optional-reranking)  
9. [Prompt Construction & LLM Generation](#prompt-construction--llm-generation)  
10. [Evaluation & Testing](#evaluation--testing)  
11. [Optional UI Chat Interface](#optional-ui-chat-interface)  
12. [Scaling & Production Tips](#scaling--production-tips)  
13. [Contributing](#contributing)  
14. [License](#license)  

---

## Features

- Upload books or PDFs for AI-based question answering  
- OCR support for scanned documents  
- Smart chunking with overlap  
- Sentence-transformer embeddings stored in a vector database  
- Semantic retrieval + optional cross-encoder reranking  
- LLM answer generation with page-level citations  
- Persistent storage on Google Drive or local filesystem  
- Optional Streamlit chat UI  

---

## Tech Stack

| Component | Library / Tool |
|-----------|----------------|
| PDF Parsing | pdfplumber, pytesseract (OCR) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | Chroma (DuckDB+Parquet persistence) |
| Retrieval & Reranking | Chroma semantic search, Cross-Encoder (optional) |
| LLM | OpenAI GPT-4/GPT-3.5, HuggingFace Transformers (optional) |
| UI (Optional) | Streamlit + ngrok |

---

## Setup & Installation

1. Open **Google Colab** and create a new notebook.  
2. Install dependencies:

```bash
!pip install -q pdfplumber sentence-transformers chromadb faiss-cpu transformers \
             langchain tiktoken openai pytesseract pillow python-magic
````

3. (Optional) Mount Google Drive to persist embeddings and PDFs:

```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_BASE = "/content/drive/MyDrive/rag_books"
import os
os.makedirs(DRIVE_BASE, exist_ok=True)
```

---

## PDF Ingestion

**Option A — Upload from local machine:**

```python
from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]
```

**Option B — Load from Drive:**

```python
pdf_path = os.path.join(DRIVE_BASE, "mybook.pdf")
assert os.path.exists(pdf_path), "PDF not found!"
```

---

## Text Extraction (with OCR fallback)

```python
import pdfplumber
from PIL import Image
import pytesseract

def extract_text_with_ocr(pdf_path, ocr_lang='eng'):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text, "is_ocr": False})
            else:
                pil = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil, lang=ocr_lang)
                pages.append({"page": i, "text": ocr_text, "is_ocr": True})
    return pages

pages = extract_text_with_ocr(pdf_path)
```

---

## Chunking & Metadata

```python
def chunk_text_pages(pages, max_words=450, overlap_words=100):
    chunks = []
    for p in pages:
        words = p['text'].split()
        i, chunk_id = 0, 0
        while i < len(words):
            chunk_words = words[i:i+max_words]
            chunk_text = " ".join(chunk_words).strip()
            if chunk_text:
                meta = {"page": p["page"], "chunk_id": chunk_id, "is_ocr": p["is_ocr"]}
                chunks.append({"text": chunk_text, "meta": meta})
            i += max_words - overlap_words
            chunk_id += 1
    return chunks

chunks = chunk_text_pages(pages)
```

---

## Embeddings & Vector Store

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

EMBED_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = os.path.join(DRIVE_BASE, "chroma_db")
COLLECTION_NAME = "books"

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))
collection = client.get_or_create_collection(COLLECTION_NAME)

ids, docs, metadatas, embeddings = [], [], [], []
for c in chunks:
    uid = str(uuid.uuid4())
    ids.append(uid)
    docs.append(c['text'])
    meta = c['meta'].copy()
    meta['source_file'] = os.path.basename(pdf_path)
    metadatas.append(meta)
    embeddings.append(embedder.encode(c['text']).tolist())

collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
client.persist()
```

---

## Retrieval & Optional Reranking

```python
# Simple retrieval
def retrieve(query, top_k=6):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents","metadatas","distances"])
    hits = [{"text": d, "meta": m, "distance": dist} for d, m, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0])]
    return hits
```

Optional reranking:

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_and_rerank(query, top_k=20, rerank_k=5):
    hits = retrieve(query, top_k=top_k)
    candidates = [h['text'] for h in hits]
    scores = reranker.predict([(query, c) for c in candidates])
    for i, s in enumerate(scores): hits[i]['rerank_score'] = float(s)
    return sorted(hits, key=lambda x: x['rerank_score'], reverse=True)[:rerank_k]
```

---

## Prompt Construction & LLM Generation

```python
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

def build_context_snippets(hits, max_chars=1500):
    return "\n\n---\n\n".join(
        f"[source: {h['meta'].get('source_file')} | page: {h['meta'].get('page')}]\n{h['text'][:max_chars]}"
        for h in hits
    )

question = "Summarize chapter 3."
hits = retrieve_and_rerank(question, top_k=25, rerank_k=8)
context = build_context_snippets(hits)

prompt = f"""
Use ONLY the context below to answer. If not in context, say "Not enough information."

CONTEXT:
{context}

QUESTION: {question}
"""

resp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role":"user","content":prompt}],
    max_tokens=400,
    temperature=0
)
answer = resp['choices'][0]['message']['content'].strip()
print(answer)
```

---

## Evaluation & Testing

* **Precision@k**: check if gold page is in top-k results
* **Hallucination test**: ask questions with no answer; system should respond `"Not enough information"`
* **Human review**: verify answers match cited pages

---

## Optional UI Chat Interface

```python
import streamlit as st

st.title("PDF RAG Chat")
question = st.text_input("Ask a question:")
if st.button("Ask"):
    hits = retrieve_and_rerank(question, top_k=25, rerank_k=8)
    context = build_context_snippets(hits)
    answer = call_llm_answer(question, context)  # define function for LLM call
    st.write("**Answer:**", answer)
```

Run Streamlit locally or in Colab with ngrok.

---

## Scaling & Production Tips

* Persist embeddings to Drive or cloud storage
* Hybrid search (BM25 + dense) for higher recall
* Cross-encoder reranking for better relevance
* Trim context to fit LLM token limits
* Store metadata for provenance
* Monitor queries and outputs for hallucinations

---

## Contributing

PRs, feature suggestions, and bug reports welcome.
Focus areas: improved chunking, evaluation metrics, multi-PDF support, UI enhancements.

---

## License

MIT License – free to use, modify, and share.

```

---



