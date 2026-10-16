"""冒烟测试：不经过 MCP 协议，直接调用检索服务，验证「知识库 → 索引 → 检索」链路可用。

用途：
1. 装完依赖后确认环境 OK（模型能加载、缓存能命中）；
2. 排查问题时把「检索逻辑错」和「MCP 协议层错」分开定位 —— 这是本项目的基本调试纪律。

运行：
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

import json
import sys
from pathlib import Path

# 允许以 `python scripts/smoke_test.py` 方式直接运行（把仓库根目录加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  # noqa: E402
from rag.service import KnowledgeService  # noqa: E402

# 三条探针问题：前两条应召回到对应文档，第三条故意问知识库里没有的内容（看降级行为）
PROBE_QUERIES = [
    "Transformer 的自注意力机制是怎么计算的？",
    "LoRA 微调的原理和典型参数是什么？",
    "今天武汉的天气怎么样？",
]


def main() -> int:
    print("[1/3] 加载配置并启动检索服务 ...")
    config = load_config()
    print(f"      配置文件：{config['_config_path']}")
    print(f"      知识库：  {config['knowledge_base']['path']}")

    service = KnowledgeService(config)

    print("[2/3] 构建索引（首次运行会算向量并写 MD5 缓存）...")
    service.build_index()
    stats = service.stats()
    print(
        f"      文档 {stats['documents']} 篇 → chunk {stats['chunks']} 个 | "
        f"模型 {stats['embedding_model']} ({stats['embedding_device']}, {stats['vector_dim']} 维) | "
        f"建索引 {stats['index_build_seconds']}s | 缓存命/未命中 {stats['cache_hits']}/{stats['cache_misses']}"
    )

    print("[3/3] 逐条检索（每条只返回 score >= 阈值的片段）...\n")
    for i, query in enumerate(PROBE_QUERIES, 1):
        payload = service.search(query, top_k=3)
        print(f"--- 探针 {i}：{query}")
        print(
            f"    matched={payload['matched']} "
            f"filtered_by_threshold={payload['filtered_by_threshold']} "
            f"elapsed={payload['elapsed_ms']}ms"
        )
        if payload["matched"] == 0:
            print(f"    hint: {payload.get('hint')}")
        for r in payload["results"]:
            preview = r["text"].replace("\n", " ")[:60]
            print(f"    rank={r['rank']} score={r['score']} source={r['source']} | {preview}...")
        print()

    print("完整结构化返回（以第 1 条探针为例）：")
    print(json.dumps(service.search(PROBE_QUERIES[0], top_k=2), ensure_ascii=False, indent=2)[:1200])
    print("\n[OK] 检索链路可用（模型加载 + 索引 + 检索 + 阈值过滤 + 空结果降级）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
