"""检索：NumPy 余弦相似度 + 相似度阈值过滤（最小移植版）。

与 rag-eval-platform 的差异：
- 返回结构里额外带出 `filtered_by_threshold`（被阈值丢弃的候选数），
  让调用方能区分「知识库里没有」和「有但分数太低」——这是 MCP 工具可靠性的关键信息；
- 其余（余弦算法、阈值语义、排序口径）与原平台一致。
"""

from typing import Dict, List, Tuple

import numpy as np

from rag.logging_utils import get_logger

logger = get_logger()


class RetrievalEngine:
    """检索引擎：向量余弦搜索 + 相似度阈值过滤。"""

    def __init__(self, config: Dict):
        ret_cfg = config["retrieval"]
        self.top_k = int(ret_cfg["top_k"])
        self.similarity_threshold = float(ret_cfg["similarity_threshold"])
        self.max_top_k = int(ret_cfg.get("max_top_k", 10))

    def search(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        chunks: List[Dict],
        top_k: int | None = None,
    ) -> Tuple[List[Dict], int]:
        """返回 (结果列表, 被阈值丢弃的候选数)。

        结果结构：[{..., chunk 元数据..., "score": 0.71, "retrieved_rank": 1}, ...]

        为什么是暴力余弦：知识库规模是几十~几百 chunk（本仓库样例 5 篇文档），
        矩阵乘法在这个量级完全不是瓶颈，换来零额外依赖、行为完全可解释。
        规模上量后（万级以上 chunk）再换 Faiss / 向量库，接口不用改。
        """
        k = self.top_k if top_k is None else max(1, min(int(top_k), self.max_top_k))

        # 归一化向量下，点积 == 余弦相似度
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores = (query_embedding @ doc_embeddings.T).flatten()

        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict] = []
        filtered = 0
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.similarity_threshold:
                filtered += 1
                continue  # 低分候选直接丢弃：宁可不答，也不给模型喂不相关资料
            results.append(
                {
                    **chunks[idx],
                    "score": round(score, 4),
                    "retrieved_rank": len(results) + 1,
                }
            )

        logger.info(f"检索完成：{len(results)}/{k} 条结果（阈值={self.similarity_threshold}，丢弃={filtered}）")
        return results, filtered
