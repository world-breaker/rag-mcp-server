"""知识库检索服务：加载文档 → 切分 → 向量化 → 建索引 → 检索。

这是「检索能力」的唯一实现处：MCP 工具层、冒烟脚本、pytest 都调用它，
保证协议层与核心逻辑解耦（换传输协议不用改检索代码，反之亦然）。
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from rag.chunking import FixedSizeChunker
from rag.config import load_config
from rag.embedding import EmbeddingEngine
from rag.logging_utils import get_logger
from rag.retrieval import RetrievalEngine

logger = get_logger()


class KnowledgeService:
    """进程内单例风格服务：索引懒加载一次，之后的检索都复用内存里的向量矩阵。"""

    def __init__(self, config: Dict | None = None):
        self.config = config or load_config()
        self.chunker = FixedSizeChunker(self.config)
        self.retriever = RetrievalEngine(self.config)
        self.embedder: EmbeddingEngine | None = None  # 懒加载：不检索就不加载模型

        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []
        self.doc_embeddings = None

        self.build_count = 0
        self.build_seconds: float | None = None
        self.search_count = 0

        # 建索引单飞锁：HTTP 下多个请求可能同时发现「索引还没建」，
        # 没有锁就会并发建 N 次（重复加载模型/重复编码）。有了锁只建一次，其余线程等它建完直接复用。
        self._build_lock = threading.Lock()

    # ========== 文档加载 ==========

    def _load_documents(self) -> List[Dict]:
        """支持三种来源：单个 .json（[{text, source}]）、目录（.txt/.json）、单个 .txt/.md。"""
        path = Path(self.config["knowledge_base"]["path"])
        if not path.exists():
            raise FileNotFoundError(f"知识库路径不存在：{path}（请检查 config.yaml 的 knowledge_base.path）")

        if path.is_dir():
            docs: List[Dict] = []
            for file in sorted(path.glob("*.txt")) + sorted(path.glob("*.md")):
                docs.append({"text": file.read_text(encoding="utf-8"), "source": file.name})
            for file in sorted(path.glob("*.json")):
                docs.extend(self._read_json(file))
            logger.info(f"从目录加载了 {len(docs)} 个文档：{path}")
            return docs

        if path.suffix.lower() == ".json":
            docs = self._read_json(path)
            logger.info(f"从 JSON 加载了 {len(docs)} 个文档：{path.name}")
            return docs

        logger.info(f"从单文件加载了 1 个文档：{path.name}")
        return [{"text": path.read_text(encoding="utf-8"), "source": path.name}]

    @staticmethod
    def _read_json(path: Path) -> List[Dict]:
        """JSON 支持两种形态：[{text,source}, ...] 或 {"text": ..., "source": ...}。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"无法识别的知识库 JSON 结构：{path}（应为列表或对象）")

    # ========== 建索引 ==========

    def build_index(self) -> None:
        """加载文档 → 切分 → 向量化。重复调用会重建（幂等）。"""
        start = time.perf_counter()
        self.documents = self._load_documents()
        if not self.documents:
            raise ValueError("知识库为空：没有可建索引的文档")

        all_chunks: List[Dict] = []
        for doc in self.documents:
            doc = dict(doc)  # 浅拷贝：不改动调用方数据
            text = doc.pop("text")
            meta = doc  # 其余字段（source 等）作为元数据挂在每个 chunk 上
            source = meta.get("source", f"doc{len(all_chunks)}")
            meta["source"] = source
            all_chunks.extend(self.chunker.chunk(text, meta))

        # 全局唯一 chunk_index：原平台同样处理（各文档各自从 0 编号会重复，
        # 检索结果需要能被上层唯一标识/引用）
        for global_idx, chunk in enumerate(all_chunks):
            chunk["doc_chunk_index"] = chunk["chunk_index"]
            chunk["chunk_index"] = global_idx

        self.chunks = all_chunks
        if self.embedder is None:
            self.embedder = EmbeddingEngine(self.config)
        self.doc_embeddings = self.embedder.encode(all_chunks)

        self.build_count += 1
        self.build_seconds = round(time.perf_counter() - start, 3)
        logger.info(
            f"索引构建完成：{len(self.documents)} 篇文档 → {len(self.chunks)} 个 chunk，"
            f"维度 {self.doc_embeddings.shape[1]}，耗时 {self.build_seconds}s"
        )

    def _ensure_index(self) -> None:
        """懒加载索引（单飞）：只有第一个发现「还没建」的线程去建，其余线程等它建完复用。"""
        if self.index_ready():
            return
        with self._build_lock:
            # 双检：等锁期间可能已经被别的线程建好了
            if not self.index_ready():
                self.build_index()

    def index_ready(self) -> bool:
        """只读判断：索引是否已就绪（不触发任何构建；/healthz 也用它）。"""
        return self.doc_embeddings is not None and bool(self.chunks)

    def warmup(self) -> None:
        """服务启动时预热：把「模型加载 + 建索引」的耗时挪到启动阶段，
        避免第一次工具调用的延迟过大（MCP 客户端通常有超时）。"""
        started = time.perf_counter()
        self._ensure_index()
        logger.info(f"预热完成，耗时 {round(time.perf_counter() - started, 3)}s")

    # ========== 检索 ==========

    def search(self, query: str, top_k: int | None = None) -> Dict[str, Any]:
        """检索知识库，返回结构化结果（含分数与被阈值过滤的数量）。

        注意：这里不抛「没检索到」的异常——空结果是一种正常业务结果，
        用结构字段（matched / filtered_by_threshold / hint）表达原因，
        让上层 Agent 能区分「知识 없음」与「工具坏了」两种失败。
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query 不能为空")

        self._ensure_index()
        self.search_count += 1

        k = int(top_k) if top_k is not None else int(self.config["retrieval"]["top_k"])
        k_effective = max(1, min(k, self.retriever.max_top_k))

        started = time.perf_counter()
        query_emb = self.embedder.encode([{"text": query}])
        results, filtered = self.retriever.search(
            query_emb, self.doc_embeddings, self.chunks, top_k=k_effective
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        payload: Dict[str, Any] = {
            "query": query,
            "top_k": k_effective,
            "top_k_truncated": k_effective != k,  # 被 max_top_k 上限截断
            "similarity_threshold": self.retriever.similarity_threshold,
            "matched": len(results),
            "filtered_by_threshold": filtered,
            "elapsed_ms": elapsed_ms,
            "results": [
                {
                    "rank": r["retrieved_rank"],
                    "score": r["score"],
                    "source": r.get("source", "unknown"),
                    "chunk_index": r.get("chunk_index"),
                    "text": r["text"],
                }
                for r in results
            ],
        }
        if not results:
            payload["hint"] = (
                "知识库中没有相似度高于阈值的内容："
                f"本次候选 {filtered} 条因分数 < {self.retriever.similarity_threshold} 被过滤。"
                "可换一种问法，或改用知识库里的术语重述问题。"
            )
        return payload

    # ========== 元信息 ==========

    def stats(self) -> Dict[str, Any]:
        """索引元信息：知识库范围、模型、缓存命中情况——供调用方（和人）判断能力边界。"""
        self._ensure_index()
        sources = sorted({c.get("source", "unknown") for c in self.chunks})
        return {
            "knowledge_base_path": self.config["knowledge_base"]["path"],
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "sources": sources,
            "embedding_model": self.config["embedding"]["model_name"],
            "embedding_device": self.embedder.device,
            "vector_dim": int(self.doc_embeddings.shape[1]),
            "chunk_size": self.config["chunking"]["chunk_size"],
            "chunk_overlap": self.config["chunking"]["chunk_overlap"],
            "similarity_threshold": self.retriever.similarity_threshold,
            "default_top_k": self.retriever.top_k,
            "max_top_k": self.retriever.max_top_k,
            "cache_hits": self.embedder.cache_hits,
            "cache_misses": self.embedder.cache_misses,
            "index_build_count": self.build_count,
            "index_build_seconds": self.build_seconds,
            "search_count": self.search_count,
        }
