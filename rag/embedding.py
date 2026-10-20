"""向量化：BGE-small-zh-v1.5 + （模型名+文本）MD5 内容缓存（最小移植版）。

与 rag-eval-platform 的差异：
- device 支持 "auto"（有 CUDA 用 GPU，否则 CPU），原平台写死单一设备；
- 新增 offline 开关与缓存命中计数（供 stats 工具展示）；
- 归一化方式、模型与维度与原平台完全一致，两个仓库产出的向量可互换；
- **缓存键比原平台多带一个模型名**（见下），因此两边的缓存文件**不通用**（向量通用、缓存不通用）；
- 缓存读写加了**并发保护**（原子落盘 + 同 key 单飞 + 坏文件自愈）：
  HTTP 传输下工具函数由 SDK 放在线程池里执行（`anyio.to_thread.run_sync`），
  并发冷启动时会多线程同时写同一个 `.npy` —— 压测实测会写出半截文件
  （`Failed to read all data for array ... file seems not fully written?`），
  stdio 单客户端模式下永远碰不到这个问题。

## 缓存键为什么必须带模型名

缓存里存的是「**某个模型**算出来的向量」。如果缓存键只按文本取（早期实现），
那么改掉 `embedding.model_name` 之后旧缓存**照样命中** —— 拿回来的是**旧模型**的向量，
但检索结果/指标看上去「一切正常」。这是典型的**静默污染**：实验数字看着对，
实际跑的是另一个模型。带上映射模型名之后，换模型 = 换键 = 缓存自动失效并重算
（代价只是第一次重算一遍向量，见 README §5.9）。
"""

import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rag.config import resolve_device
from rag.logging_utils import get_logger

logger = get_logger()

#: 单飞锁表上限：超出后整体清空。清空只会让「同 key 并发」退化为一同重算（写入仍是原子的），
#: 不会破坏正确性，因此这里用最简单的方式给内存封顶
MAX_KEY_LOCKS = 4096

#: 模型名 → 文件名安全短标识：只保留这些字符，其余换成 "_"
_MODEL_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def model_slug(model_name: str) -> str:
    """把模型名变成文件名安全的短标识（'BAAI/bge-small-zh-v1.5' → 'BAAI_bge-small-zh-v1.5'）。

    只为让缓存文件名**人可读**（一眼看出是哪份模型的向量）；
    真正决定缓存身份的是 `md5(模型名 + 文本)`，所以两个不同模型即使 slug 撞了也不会互相污染。
    """
    slug = _MODEL_SLUG_RE.sub("_", str(model_name)).strip("_")
    if not slug:
        slug = "model"
    return slug[:40]


class EmbeddingEngine:
    """向量化引擎：模型封装 + 基于（模型名+文本）MD5 的离线缓存（文档不变不重算）。"""

    def __init__(self, config: Dict, model=None):
        """`model` 可注入一个已加载好的编码器（单测用假模型即可，不必下载 BGE）。

        生产路径不传 `model`：照常按 `embedding.model_name` 加载 SentenceTransformer。
        """
        emb_cfg = config["embedding"]
        self.model_name = emb_cfg["model_name"]
        self.device = resolve_device(config)
        self.batch_size = int(emb_cfg["batch_size"])
        self.cache_dir = Path(emb_cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = bool(emb_cfg.get("offline", True))

        self.cache_hits = 0
        self.cache_misses = 0

        # 保护「锁表 + 命中/未命中计数」：计数在 HTTP 并发下会被多线程同时自增
        self._state_lock = threading.Lock()
        self._key_locks: Dict[str, threading.Lock] = {}

        if self.offline:
            # 模型已在本机缓存 → 禁止联网：既快，又保证复现时不被网络状态影响
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if model is not None:
            # 单测注入：让「缓存键隔离」这类用例离线可跑（CI runner 上没有 BGE 缓存）
            self.model = model
            logger.info(f"Embedding 模型由调用方注入：{self.model_name}")
            return

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

    # ========== 缓存键 ==========

    def _cache_key(self, texts: List[str]) -> str:
        """缓存键：**模型名 + 文本内容**的 MD5。

        为什么必须带模型名：缓存里存的是「某个模型算出来的向量」。
        早期实现只按文本取键，改了 `embedding.model_name` 之后旧缓存照样命中 ——
        拿回来的是**旧模型**的向量，指标却看着「正常」（静默污染）。
        带上模型名之后：换模型 = 换键 = 旧缓存自动失效并重算。

        文本之间用 "|||" 拼接（顺序不同 = 不同键，避免错配）。
        """
        payload = f"{self.model_name}\n" + "|||".join(texts)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, texts: List[str]) -> Path:
        """缓存文件路径：`<模型标识>-<内容哈希>.npy`（模型标识让人一眼看出是哪份模型的向量）。"""
        return self.cache_dir / f"{model_slug(self.model_name)}-{self._cache_key(texts)}.npy"

    # ========== 缓存读写（并发安全） ==========

    def _key_lock(self, cache_key: str) -> threading.Lock:
        """取（或建）某个缓存键的单飞锁：保证同一 key 只被算一次。"""
        with self._state_lock:
            if len(self._key_locks) > MAX_KEY_LOCKS:
                self._key_locks.clear()
            lock = self._key_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[cache_key] = lock
            return lock

    def _count(self, field: str) -> None:
        with self._state_lock:
            setattr(self, field, getattr(self, field) + 1)

    def _load_cache(self, cache_path: Path, expected_rows: int) -> Optional[np.ndarray]:
        """读缓存；文件缺失/半截/形状不符都按「未命中」处理（自愈，不抛异常给调用方）。"""
        if not cache_path.exists():
            return None
        try:
            array = np.load(cache_path)
        except Exception as exc:
            # 历史残留的半截文件（旧版本非原子写入 / 进程被 kill）→ 重算并覆盖
            logger.warning(
                f"Embedding 缓存不可读，按未命中处理（将重算覆盖）：{cache_path.name}"
                f"（{type(exc).__name__}: {exc}）"
            )
            return None
        if array.shape[0] != expected_rows:
            logger.warning(
                f"Embedding 缓存形状不符（{array.shape} != ({expected_rows}, ...)），按未命中处理：{cache_path.name}"
            )
            return None
        return array

    def _save_cache(self, cache_path: Path, embeddings: np.ndarray) -> None:
        """原子落盘：先写进程/线程唯一的临时文件，再 `os.replace` 到目标路径。

        直接 `np.save(目标路径)` 会让并发读者看到「文件已存在但内容没写完」的中间态
        （Windows/NTFS 上 `os.replace` 对同卷是原子替换），这是压测暴露出这个 bug 的根因。

        两处权衡：
        - `os.replace` 在 Windows 上可能因为目标文件被瞬间占用而抛 `WinError 5` → 重试几次；
        - 缓存写失败**不影响本次检索结果**（向量已经算出来了），只记一条 warning 后继续，
          下次同样内容再重算一次即可 —— 「缓存是优化，不该成为故障点」。
        """
        tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        last_error: Exception | None = None
        try:
            with open(tmp_path, "wb") as handle:
                np.save(handle, embeddings)
            for attempt in range(3):
                try:
                    os.replace(tmp_path, cache_path)
                    return
                except OSError as exc:  # Windows：目标瞬间被占用（WinError 5）→ 退避重试
                    last_error = exc
                    time.sleep(0.02 * (attempt + 1))
            logger.warning(
                f"Embedding 缓存写入失败，本次检索不受影响（下次会重算）：{cache_path.name}（{last_error}）"
            )
        except Exception as exc:  # 磁盘满/权限等问题都不该让检索失败
            logger.warning(f"Embedding 缓存写入异常，本次检索不受影响：{cache_path.name}（{type(exc).__name__}: {exc}）")
        finally:
            tmp_path.unlink(missing_ok=True)  # 失败时不留垃圾文件

    def encode(self, chunks: List[Dict]) -> np.ndarray:
        """把 chunk 列表转成向量矩阵；命中缓存则跳过模型计算。"""
        texts = [c["text"] for c in chunks]
        cache_key = self._cache_key(texts)
        cache_path = self._cache_path(texts)

        cached = self._load_cache(cache_path, expected_rows=len(texts))
        if cached is not None:
            self._count("cache_hits")
            logger.info(f"Embedding 缓存命中 → {cache_path.name}")
            return cached

        # 单飞：同一 key 只让一个线程算，其余线程等它写完直接读缓存
        with self._key_lock(cache_key):
            cached = self._load_cache(cache_path, expected_rows=len(texts))
            if cached is not None:
                self._count("cache_hits")
                logger.info(f"Embedding 缓存命中（复用同 key 线程的计算结果）→ {cache_path.name}")
                return cached

            self._count("cache_misses")
            logger.info(f"计算 Embedding：{len(texts)} 条文本（模型={self.model_name}）")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,  # 归一化后「点积 == 余弦相似度」
            )
            self._save_cache(cache_path, embeddings)
            logger.info(f"Embedding 已缓存 → {cache_path.name}")
            return embeddings
