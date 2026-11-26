#!/usr/bin/env python3
"""app.py —— 工脉 v0.1 服务端（FastAPI）

启动：
    python app.py            # http://127.0.0.1:8000
    # 或 uvicorn app:app --host 127.0.0.1 --port 8000

接口：
    GET  /                   前端单页
    GET  /api/stats          知识库统计
    GET  /api/docs           文档列表
    GET  /api/docs/{doc_id}  文档章节与页码内容
    POST /api/ask            证据化问答
    GET  /api/graph          知识图谱节点与边
    GET  /api/diagnose       巡检 -> 诊断
    POST /api/rebuild        重新解析并建索引
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import core

BASE = Path(__file__).resolve().parent
WEB = BASE / "web"

app = FastAPI(title="工脉 · 工业设备知识中枢 v0.1", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    # 首次启动自动建索引；已建过则复用（除非显式 rebuild）
    if not core.INDEX_PATH.is_file() or not core.ENGINE.chunks:
        core.ENGINE.build()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=300)
    top_k: int = Field(default=5, ge=1, le=10)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/web/{name}")
def static(name: str) -> FileResponse:
    path = (WEB / name).resolve()
    if not str(path).startswith(str(WEB.resolve())) or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/stats")
def stats() -> dict:
    return core.ENGINE.stats()


@app.get("/api/docs")
def docs() -> dict:
    return {"docs": core.ENGINE.docs}


@app.get("/api/docs/{doc_id}")
def doc_detail(doc_id: str) -> dict:
    chunks = [c for c in core.ENGINE.chunks if c.doc_id == doc_id]
    if not chunks:
        return {"error": "not found"}
    return {
        "doc_id": doc_id,
        "title": chunks[0].doc_title,
        "type": chunks[0].doc_type,
        "chunks": [
            {
                "page": c.page,
                "section": c.section,
                "text": c.text[:1200],
            }
            for c in chunks
        ],
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    return core.ENGINE.answer(req.question, top_k=req.top_k)


@app.get("/api/graph")
def graph() -> dict:
    return core.ENGINE.graph


@app.get("/api/diagnose")
def diagnose() -> dict:
    return core.ENGINE.diagnose()


@app.post("/api/rebuild")
def rebuild() -> dict:
    return core.ENGINE.build()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
