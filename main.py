from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import utilities as ut

VECTORSTORE_PATH = "./chroma_db"

app = FastAPI(title="RAG API")

class QueryRequest(BaseModel):
    query: str

class Reference(BaseModel):
    source: str
    page: int

class QueryResponse(BaseModel):
    summary: str
    references: list[Reference]
    related_images: list[str]

@app.post("/rag/query", response_model=QueryResponse)
def run_rag_query(req: QueryRequest):

    docs, summary = ut.semantic_search(req.query, VECTORSTORE_PATH)

    if isinstance(summary, str) and summary.startswith("⚠️"):
        raise HTTPException(status_code=400, detail=summary)

    # ---- references ----
    refs = set()
    for doc in docs:
        meta = doc.metadata or {}
        refs.add((meta.get("source"), meta.get("page")))

    references = [
        {"source": src, "page": page}
        for src, page in sorted(refs)
    ]

    # ---- images ----
    images = ut.get_all_images(docs)

    return {
        "summary": summary,
        "references": references,
        "related_images": images
    }

