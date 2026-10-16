"""pytest：检索服务的核心行为断言（需要本机已缓存 BGE 模型，CPU 下约几秒）。

运行：
    .venv\\Scripts\\python.exe -m pytest tests -v
"""

import sys
from pathlib import Path

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
