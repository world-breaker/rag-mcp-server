"""pytest：检索服务的核心行为断言（需要本机已缓存 BGE 模型，CPU 下约几秒）。

运行：
    .venv\\Scripts\\python.exe -m pytest tests -v
"""

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  # noqa: E402
from rag.service import KnowledgeService  # noqa: E402


@pytest.fixture(scope="module")
def service() -> KnowledgeService:
    config = load_config()
    config["embedding"]["device"] = "cpu"  # 测试固定 CPU，避免不同机器行为差异
    svc = KnowledgeService(config)
    svc.build_index()
    return svc


def test_index_stats(service):
    stats = service.stats()
    assert stats["documents"] == 5
    assert stats["chunks"] >= 5
    assert stats["vector_dim"] == 512
    assert stats["similarity_threshold"] == 0.3


def test_search_returns_ranked_results(service):
    payload = service.search("Transformer 的自注意力怎么算？", top_k=3)
    assert payload["matched"] >= 1
    scores = [r["score"] for r in payload["results"]]
    assert scores == sorted(scores, reverse=True)  # 分数降序
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["source"].endswith(".txt")


def test_top_k_is_capped(service):
    payload = service.search("RAG 的检索优化", top_k=999)
    assert payload["top_k"] == service.retriever.max_top_k
    assert payload["top_k_truncated"] is True
    assert payload["matched"] <= service.retriever.max_top_k


def test_out_of_domain_query_is_filtered(service):
    """领域外问题应被阈值过滤掉，而不是硬返回不相关片段。"""
    payload = service.search("明天武汉的天气怎么样？", top_k=3)
    assert payload["matched"] == 0
    assert payload["filtered_by_threshold"] > 0
    assert "hint" in payload


def test_empty_query_rejected(service):
    with pytest.raises(ValueError):
        service.search("   ")


def test_result_is_json_serializable(service):
    import json

    payload = service.search("LoRA 的原理", top_k=2)
    text = json.dumps(payload, ensure_ascii=False)
    assert "results" in json.loads(text)


# ========== 并发安全（HTTP 传输下工具函数跑在线程池里，必须扛住真并发） ==========

def test_concurrent_queries_compute_embedding_once_and_write_cache_atomically(service, tmp_path):
    """8 个线程同时冷启动同一个 query：只算 1 次、只留 1 个完整缓存文件。

    背景（HTTP 压测实测出来的 bug）：旧版直接 `np.save(目标路径)`，并发时会写出半截
    `.npy`（`Failed to read all data for array ... file seems not fully written?`），
    之后所有读到该文件的请求都报错。现在改为「同 key 单飞 + 临时文件 + os.replace 原子替换」。
    """
    engine = service.embedder  # 模型已由 fixture 加载，这里只测缓存层的并发行为
    original_cache_dir = engine.cache_dir
    engine.cache_dir = tmp_path
    probe = [{"text": f"并发写入探针 {uuid.uuid4()}"}]  # 每次唯一 → 必定未命中
    try:
        misses_before = engine.cache_misses
        with ThreadPoolExecutor(max_workers=8) as pool:
            vectors = list(pool.map(lambda _: engine.encode(probe), range(8)))

        assert engine.cache_misses - misses_before == 1  # 单飞：只算了一次
        assert all(vector.shape == (1, 512) for vector in vectors)
        assert np.allclose(vectors[0], vectors[7])  # 所有线程拿到的是同一份结果

        files = [path.name for path in tmp_path.iterdir()]
        assert len(files) == 1 and files[0].endswith(".npy"), files  # 无残留 .tmp，只落一个文件
    finally:
        engine.cache_dir = original_cache_dir


def test_concurrent_first_search_builds_index_only_once(service):
    """索引未就绪时 8 个线程同时进来：只允许建一次（否则重复加载模型/重复编码）。"""
    original_state = (service.doc_embeddings, service.chunks)
    service.doc_embeddings, service.chunks = None, []  # 模拟「冷启动」
    builds_before = service.build_count
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: service.search("LoRA 的原理", top_k=1), range(8)))

        assert service.build_count - builds_before == 1
        assert service.index_ready() is True
    finally:
        if not service.index_ready():  # 出错时恢复原状态，不影响后续测试
            service.doc_embeddings, service.chunks = original_state
