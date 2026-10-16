"""向量化：BGE-small-zh-v1.5 + 文本 MD5 内容缓存（最小移植版）。

与 rag-eval-platform 的差异：
- device 支持 "auto"（有 CUDA 用 GPU，否则 CPU），原平台写死单一设备；
- 新增 offline 开关与缓存命中计数（供 stats 工具展示）；
- 缓存键算法、归一化方式、模型与维度与原平台完全一致，两个仓库产出的向量可互换。
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from rag.config import resolve_device
from rag.logging_utils import get_logger

logger = get_logger()


class EmbeddingEngine:
    """向量化引擎：模型封装 + 基于文本 MD5 的离线缓存（文档不变不重算）。"""

    def __init__(self, config: Dict):
        emb_cfg = config["embedding"]
        self.model_name = emb_cfg["model_name"]
        self.device = resolve_device(config)
        self.batch_size = int(emb_cfg["batch_size"])
        self.cache_dir = Path(emb_cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = bool(emb_cfg.get("offline", True))

        self.cache_hits = 0
        self.cache_misses = 0

        if self.offline:
            # 模型已在本机缓存 → 禁止联网：既快，又保证复现时不被网络状态影响
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # 延迟导入：只有真正要算向量时才付模型加载的代价
        from sentence_transformers import SentenceTransformer

        logger.info(f"加载 Embedding 模型：{self.model_name}（device={self.device}, offline={self.offline}）")
        try:
            self.model = SentenceTransformer(
                self.model_name, device=self.device, local_files_only=self.offline
            )
        except TypeError:
            # 兼容不支持 local_files_only 参数的旧版本
            self.model = SentenceTransformer(self.model_name, device=self.device)
        # sentence-transformers 5.x 把 get_sentence_embedding_dimension 改名为 get_embedding_dimension
        dimension = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        logger.info(f"模型加载完成，向量维度：{dimension()}")

    @staticmethod
    def _hash_texts(texts: List[str]) -> str:
        """缓存键：所有文本按顺序拼接后取 MD5（顺序不同 = 不同键，避免错配）。"""
        joined = "|||".join(texts)
        return hashlib.md5(joined.encode("utf-8")).hexdigest()

    def encode(self, chunks: List[Dict]) -> np.ndarray:
        """把 chunk 列表转成向量矩阵；命中缓存则跳过模型计算。"""
        texts = [c["text"] for c in chunks]
        cache_key = self._hash_texts(texts)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            self.cache_hits += 1
            logger.info(f"Embedding 缓存命中 → {cache_path.name}")
            return np.load(cache_path)

        self.cache_misses += 1
        logger.info(f"计算 Embedding：{len(texts)} 条文本")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化后「点积 == 余弦相似度」
        )
        np.save(cache_path, embeddings)
        logger.info(f"Embedding 已缓存 → {cache_path.name}")
        return embeddings
