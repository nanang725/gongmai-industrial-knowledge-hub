#!/usr/bin/env python3
"""core.py —— 工脉 v0.1 领域引擎

职责（全部纯 Python，无重依赖，离线可跑）：
  1. 页码级解析：设备手册 PDF -> 带页码与章节的 chunk（PyMuPDF）
  2. 混合召回：中文 bigram + BM25 关键词召回，按文档多样性裁剪
  3. 实体抽取：现象 / 根因 / 措施 / 设备 / 部件 / 安全等级，每条关系带页码证据
  4. 证据化问答：强制限定在召回证据内作答，无有效证据则明确拒答
  5. 故障诊断：巡检数据 -> 停机影响分级 + 分层定位 + 根因候选

设计原则：AI 受证据约束。任何结论必须能指回「哪份文档第几页」或「哪张工单」。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MANUAL_DIR = DATA / "manuals"
WO_DIR = DATA / "workorders"
KB_DIR = DATA / "kb"
INDEX_PATH = DATA / "index.json"

# ---------------------------------------------------------------- 文本处理

STOP_CHARS = set("的了和与或是在对从把被为及等这那其之而且并但如果因为所以一个")
LATIN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\.\-/]*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
SECTION_RE = re.compile(r"(?m)^(\d+(?:\.\d+){0,2})[\s\u3000]+([^\n]{2,40})$")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens: list[str] = []
    for token in LATIN_RE.findall(text):
        if len(token) >= 2:
            tokens.append(token)
    for segment in CJK_RE.findall(text):
        if len(segment) == 1:
            if segment not in STOP_CHARS:
                tokens.append(segment)
            continue
        for i in range(len(segment) - 1):
            gram = segment[i : i + 2]
            if gram[0] in STOP_CHARS and gram[1] in STOP_CHARS:
                continue
            tokens.append(gram)
    return tokens


# ---------------------------------------------------------------- 数据结构


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    doc_type: str  # manual | workorder | knowledge
    page: int | None
    section: str
    text: str
    tokens: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    id: str
    doc_id: str
    doc_title: str
    doc_type: str
    page: int | None
    section: str
    snippet: str
    score: float


# ---------------------------------------------------------------- 解析


def parse_manual(path: Path) -> list[Chunk]:
    """页码级解析：按页读取，再按编号标题切分章节，保证每条 chunk 可回指页码。"""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    doc_id = path.stem
    title = ""
    chunks: list[Chunk] = []

    for page_no in range(doc.page_count):
        raw = doc[page_no].get_text().strip()
        if not raw:
            continue
        lines = raw.splitlines()
        if not title:
            title = lines[0].strip()
        body_lines = [ln for ln in lines[1:] if "文档编号" not in ln]
        body = "\n".join(body_lines).strip()
        if not body:
            continue

        marks = list(SECTION_RE.finditer(body))
        if not marks:
            chunks.append(
                Chunk(
                    id=f"{doc_id}-p{page_no + 1}-0",
                    doc_id=doc_id, doc_title=title, doc_type="manual",
                    page=page_no + 1, section="", text=body,
                )
            )
            continue

        for idx, mark in enumerate(marks):
            start = mark.start()
            end = marks[idx + 1].start() if idx + 1 < len(marks) else len(body)
            section_title = f"{mark.group(1)} {mark.group(2).strip()}"
            chunks.append(
                Chunk(
                    id=f"{doc_id}-p{page_no + 1}-{idx}",
                    doc_id=doc_id, doc_title=title, doc_type="manual",
                    page=page_no + 1, section=section_title,
                    text=body[start:end].strip(),
                )
            )
    doc.close()
    return chunks


def parse_markdown(path: Path, doc_type: str) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"(?m)^#\s+(.+)$", raw)
    title = title_match.group(1).strip() if title_match else path.stem
    doc_id = path.stem

    parts = re.split(r"\n(?=##\s+)", raw)
    chunks: list[Chunk] = []
    for idx, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        head = part.splitlines()[0].lstrip("# ").strip()
        if head == title:
            head = ""
        chunks.append(
            Chunk(
                id=f"{doc_id}-{idx}",
                doc_id=doc_id, doc_title=title, doc_type=doc_type,
                page=None, section=head, text=part,
            )
        )
    return chunks


# ---------------------------------------------------------------- BM25


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        self.df: Counter = Counter()
        for chunk in chunks:
            chunk.tokens = tokenize(chunk.text)
            tf = Counter(chunk.tokens)
            self.tfs.append(tf)
            self.lengths.append(sum(tf.values()) or 1)
            for term in tf:
                self.df[term] += 1
        self.avgdl = sum(self.lengths) / max(len(self.lengths), 1)
        self.n = len(chunks)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 6) -> list[tuple[Chunk, float]]:
        q_tf = Counter(tokenize(query))
        if not q_tf:
            return []
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            tf = self.tfs[i]
            dl = self.lengths[i]
            score = 0.0
            for term, qn in q_tf.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self._idf(term) * qn * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        # 文档多样性：同一文档最多保留 2 条，避免证据同源
        result: list[tuple[Chunk, float]] = []
        per_doc: Counter = Counter()
        for chunk, score in scored:
            if per_doc[chunk.doc_id] >= 2:
                continue
            per_doc[chunk.doc_id] += 1
            result.append((chunk, score))
            if len(result) >= top_k:
                break
        return result


# ---------------------------------------------------------------- 实体抽取

EQUIPMENT_TERMS = ["PLC", "变频器", "伺服", "触摸屏", "HMI", "交换机", "工业交换机",
                   "网关", "边缘网关", "光模块", "传感器", "编码器", "机器人",
                   "视觉检测", "电表", "UPS", "上位机"]
COMPONENT_TERMS = ["光模块", "上行口", "网口", "水晶头", "网线", "光纤", "跳线", "端口",
                   "电源模块", "通讯模块", "IO 模块", "端子", "背板"]
SYMPTOM_PATTERNS = [
    (r"掉站", "掉站"), (r"断线|断连|失联|离线", "失联"),
    (r"通讯(超时|中断|失败)|通信(超时|中断|失败)", "通讯中断"),
    (r"丢包", "丢包"), (r"抖动|时延升高|延迟升高", "时延升高"),
    (r"报警|告警", "设备报警"), (r"无法(建立)?连接|连接被拒", "连接失败"),
]
CAUSE_PATTERNS = [
    (r"光功率|收光", "光链路劣化"), (r"CRC|误码", "链路误码"),
    (r"连接(数|资源).{0,6}(耗尽|占满|不足|达上限)", "连接资源耗尽"),
    (r"电磁干扰|屏蔽", "电磁干扰"), (r"环路|广播风暴", "二层环路"),
    (r"VLAN|ACL|防火墙", "网络策略限制"),
    (r"站号|机架|槽号|寄存器", "组态配置错误"),
    (r"时钟(不同步|偏差|漂移)|NTP|PTP", "时钟不同步"),
    (r"温度|散热|过热", "温升异常"),
]
REMEDY_PATTERNS = [
    (r"更换|替换|换新", "更换部件"), (r"清洁|擦拭|除尘", "清洁保养"),
    (r"重(启|新启动)", "重启设备"), (r"重新(配置|组态)|改配置", "重新配置"),
    (r"隔离|下线|摘除", "隔离故障点"), (r"升级|固件", "升级固件"),
    (r"降低频率|低频|串行访问", "降低访问频率"),
]
SAFETY_RE = re.compile(r"\b(P[0-3])\b")


def extract_entities(chunks: list[Chunk]) -> dict[str, Any]:
    nodes: dict[str, set[str]] = {
        "equipment": set(), "component": set(), "symptom": set(),
        "cause": set(), "remedy": set(), "safety": set(),
        "manual": set(), "workorder": set(),
    }
    edges: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

    for chunk in chunks:
        text = chunk.text
        eqs = [t for t in EQUIPMENT_TERMS if t in text]
        comps = [t for t in COMPONENT_TERMS if t in text]
        syms = [label for pat, label in SYMPTOM_PATTERNS if re.search(pat, text)]
        causes = [label for pat, label in CAUSE_PATTERNS if re.search(pat, text)]
        rems = [label for pat, label in REMEDY_PATTERNS if re.search(pat, text)]
        safety = sorted(set(SAFETY_RE.findall(text)))

        if chunk.doc_type == "manual":
            nodes["manual"].add(chunk.doc_title)
        if chunk.doc_type == "workorder":
            nodes["workorder"].add(chunk.doc_title)
        nodes["equipment"].update(eqs)
        nodes["component"].update(comps)
        nodes["symptom"].update(syms)
        nodes["cause"].update(causes)
        nodes["remedy"].update(rems)
        nodes["safety"].update(safety)

        def add(src_type: str, src: str, rel: str, dst_type: str, dst: str, conf: float) -> None:
            key = (src_type, src, rel, dst_type, dst)
            if key in edges:
                edges[key]["weight"] += 1
                return
            edges[key] = {
                "source_type": src_type, "source": src, "relation": rel,
                "target_type": dst_type, "target": dst,
                "doc_id": chunk.doc_id, "doc_title": chunk.doc_title,
                "page": chunk.page, "evidence": text.strip().replace("\n", " ")[:100],
                "confidence": conf, "weight": 1,
            }

        for sym in syms:
            for cause in causes:
                add("symptom", sym, "由…导致", "cause", cause, 0.7)
            for level in safety:
                add("symptom", sym, "影响等级", "safety", level, 0.75)
            # 知识载体挂接：手册/工单与现象建立引用边，避免文档节点悬空
            if chunk.doc_type == "workorder":
                add("workorder", chunk.doc_title, "记录了", "symptom", sym, 0.8)
            elif chunk.doc_type == "manual":
                add("manual", chunk.doc_title, "依据", "symptom", sym, 0.75)
        for cause in causes:
            for rem in rems:
                add("cause", cause, "由…处置", "remedy", rem, 0.65)
        for eq in eqs:
            for comp in comps:
                if eq != comp:
                    add("equipment", eq, "包含部件", "component", comp, 0.55)

    return {
        "nodes": {k: sorted(v) for k, v in nodes.items()},
        "edges": sorted(edges.values(), key=lambda e: -e["weight"])[:150],
    }


# ---------------------------------------------------------------- 引擎


class Engine:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.graph: dict[str, Any] = {"nodes": {}, "edges": []}
        self.docs: list[dict[str, Any]] = []
        self.topology: dict[str, Any] = {}
        self.patrol: dict[str, Any] = {}
        self.built_at: str = ""

    # ---------- 构建 ----------

    def build(self) -> dict[str, Any]:
        chunks: list[Chunk] = []

        if MANUAL_DIR.is_dir():
            for path in sorted(MANUAL_DIR.glob("*.pdf")):
                chunks.extend(parse_manual(path))
        if WO_DIR.is_dir():
            for path in sorted(WO_DIR.glob("*.md")):
                chunks.extend(parse_markdown(path, "workorder"))
        if KB_DIR.is_dir():
            for path in sorted(KB_DIR.glob("*.md")):
                chunks.extend(parse_markdown(path, "knowledge"))

        self.chunks = chunks
        self.bm25 = BM25(chunks) if chunks else None
        self.graph = extract_entities(chunks)

        by_doc: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            info = by_doc.setdefault(
                chunk.doc_id,
                {"doc_id": chunk.doc_id, "title": chunk.doc_title,
                 "type": chunk.doc_type, "pages": 0, "chunks": 0,
                 "sections": []},
            )
            info["chunks"] += 1
            if chunk.page:
                info["pages"] = max(info["pages"], chunk.page)
            if chunk.section:
                info["sections"].append(chunk.section)
        self.docs = sorted(by_doc.values(), key=lambda d: (d["type"], d["title"]))

        if (DATA / "topology.json").is_file():
            self.topology = json.loads((DATA / "topology.json").read_text(encoding="utf-8"))
        if (DATA / "patrol-demo.json").is_file():
            self.patrol = json.loads((DATA / "patrol-demo.json").read_text(encoding="utf-8"))

        self.built_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        snapshot = {
            "built_at": self.built_at,
            "doc_count": len(self.docs),
            "chunk_count": len(chunks),
            "graph": self.graph,
            "docs": self.docs,
        }
        INDEX_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 6) -> list[Evidence]:
        if not self.bm25:
            return []
        hits = self.bm25.search(query, top_k=top_k)
        return [
            Evidence(
                id=c.id,
                doc_id=c.doc_id, doc_title=c.doc_title, doc_type=c.doc_type,
                page=c.page, section=c.section,
                snippet=c.text.strip().replace("\n", " ")[:260],
                score=round(s, 2),
            )
            for c, s in hits
        ]

    # ---------- 问答 ----------

    def answer(self, question: str, top_k: int = 5) -> dict[str, Any]:
        evidences = self.search(question, top_k=top_k)

        # 证据门控：无有效证据则明确拒答，不编造
        if not evidences or evidences[0].score < 1.5:
            return {
                "question": question,
                "answered": False,
                "answer": "现有知识库中没有检索到与该问题相关的有效证据，"
                          "按工业现场处置规范，不做无依据推断。"
                          "建议：补充对应的设备手册或历史工单后重试。",
                "evidence": [asdict(e) for e in evidences],
                "mode": "refused",
                "confidence": 0.0,
            }

        sections = [e.section for e in evidences if e.section]
        # 机理与处置仅从最相关的前 3 条证据的完整文本提取，按共现次数排序
        cause_counter: Counter = Counter()
        remedy_counter: Counter = Counter()
        full_text = {c.id: c.text for c in self.chunks}
        for ev in evidences[:3]:
            body = full_text.get(ev.id, ev.snippet)
            for pat, label in CAUSE_PATTERNS:
                if re.search(pat, body):
                    cause_counter[label] += 1
            for pat, label in REMEDY_PATTERNS:
                if re.search(pat, body):
                    remedy_counter[label] += 1
        causes = [c for c, _ in cause_counter.most_common(3)]
        rems = [r for r, _ in remedy_counter.most_common(3)]
        safety = sorted({x for ev in evidences[:3]
                         for x in SAFETY_RE.findall(full_text.get(ev.id, ev.snippet))})

        summary = (
            f"依据 {len(evidences)} 处证据（涉及 {len({e.doc_id for e in evidences})} 份文档），"
            f"该问题主要落在「{sections[0] if sections else '设备运维'}」相关章节。"
        )
        if causes:
            summary += f" 证据指向的机理包括：{'、'.join(causes)}。"
        if rems:
            summary += f" 处置方向：{'、'.join(rems)}。"

        return {
            "question": question,
            "answered": True,
            "answer": summary,
            "causes": causes,
            "remedies": rems,
            "safety": safety,
            "evidence": [asdict(e) for e in evidences],
            "mode": "rule",
            "confidence": round(min(0.95, evidences[0].score / 60 + 0.35), 2),
        }

    # ---------- 诊断 ----------

    def diagnose(self) -> dict[str, Any]:
        report = self.patrol
        assets = {a["host"]: a for a in self.topology.get("assets", [])}
        results = report.get("results", [])
        failed = [r for r in results if not r.get("ok")]
        high_loss = [r for r in results
                     if r.get("check_type") == "ping"
                     and (r.get("extra", {}).get("loss_percent") or 0) >= 20]

        failed_hosts = {r["host"] for r in failed}
        gw = set(report.get("local", {}).get("gateways", []))

        layer, headline = "L4 应用/协议层", "关键协议端口不可达，业务通道未建立"
        if failed_hosts & gw:
            layer, headline = "L1 物理层 / L2 链路层", "默认网关不可达，故障点在本机接入侧或上行链路"
        elif len(failed_hosts) >= 2:
            layer = "L2 链路层（疑似共同上游）"
            headline = "多台资产同时不可达但网关正常，故障点更可能位于共同上游的链路或配置"
        elif not failed:
            layer, headline = "-", "全部检测项通过"

        if high_loss:
            peak = max((r.get("extra", {}).get("loss_percent") or 0) for r in high_loss)
            headline += f"；同时存在高丢包（最高 {peak}%），物理层劣化风险高"

        impact = "P3"
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        for host in failed_hosts:
            level = assets.get(host, {}).get("impact", "P3")
            if order[level] < order[impact]:
                impact = level

        query = f"{headline} {layer} " + " ".join(
            f"{assets.get(r['host'], {}).get('role', '')} {r.get('detail', '')}" for r in failed[:4]
        )
        kb_evidence = [asdict(e) for e in self.search(query, top_k=4)]

        steps = [
            "确认故障节点是否处于安全链或主工艺链，若是则先隔离并上报。",
            "按分层定位逐层取证：端口状态与 CRC 计数 → 光功率 → VLAN 与环网 → 端口与连接数。",
            "核对处置依据：优先采用手册与历史工单中同型号、同现象的成功方案。",
            "处置后复跑同一份巡检，比对失败项是否清零并归档。",
        ]

        return {
            "site": self.topology.get("site", ""),
            "impact": impact,
            "layer": layer,
            "headline": headline,
            "total": len(results),
            "failed": len(failed),
            "assets": [
                {
                    "host": a["host"], "name": a.get("name"), "role": a.get("role"),
                    "protocol": a.get("protocol"), "line": a.get("line"),
                    "impact": a.get("impact"),
                    "status": next((("ok" if r.get("ok") else "fail")
                                    for r in results if r.get("host") == a["host"]), "unknown"),
                }
                for a in self.topology.get("assets", [])
            ],
            "failures": [
                {
                    "name": r.get("name"), "host": r.get("host"),
                    "check_type": r.get("check_type"), "detail": r.get("detail"),
                    "latency": r.get("latency_ms"),
                }
                for r in failed
            ],
            "evidence": [
                {
                    "name": r.get("name"), "host": r.get("host"),
                    "check_type": r.get("check_type"),
                    "ok": r.get("ok"), "latency": r.get("latency_ms"),
                    "detail": r.get("detail"),
                }
                for r in results
            ],
            "knowledge": kb_evidence,
            "steps": steps,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ---------- 统计 ----------

    def stats(self) -> dict[str, Any]:
        pages = sum(d["pages"] for d in self.docs if d["type"] == "manual")
        return {
            "docs": len(self.docs),
            "manuals": sum(1 for d in self.docs if d["type"] == "manual"),
            "workorders": sum(1 for d in self.docs if d["type"] == "workorder"),
            "knowledge": sum(1 for d in self.docs if d["type"] == "knowledge"),
            "pages": pages,
            "chunks": len(self.chunks),
            "entities": sum(len(v) for v in self.graph.get("nodes", {}).values()),
            "edges": len(self.graph.get("edges", [])),
            "built_at": self.built_at,
            "site": self.topology.get("site", ""),
        }


ENGINE = Engine()


if __name__ == "__main__":
    ENGINE.build()
    print(json.dumps(ENGINE.stats(), ensure_ascii=False, indent=2))
    print("\n--- 检索测试 ---")
    for e in ENGINE.search("PLC 端口能连上但读写超时怎么排查", top_k=4):
        print(f"  {e.doc_title} 第{e.page}页 [{e.section}]  score={e.score}")
    print("\n--- 问答测试 ---")
    a = ENGINE.answer("焊装线多台机器人同时掉站，交换机 CRC 计数激增")
    print(a["answer"])
    print("机理:", a["causes"], "处置:", a["remedies"], "等级:", a["safety"])
    man_edges = [e for e in ENGINE.graph["edges"] if e["source_type"] in ("manual", "workorder")]
    print(f"\n文档挂接边: {len(man_edges)} 条")
