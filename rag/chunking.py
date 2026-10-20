"""文档切分：固定窗口 + 重叠（最小移植版）。

与 rag-eval-platform 的差异：
- 原平台有 fixed_size / recursive / semantic 三种策略；本仓库只移植 fixed_size
  （平台默认策略），因为 MCP Server 的定位是「检索能力对外暴露」，
  切分策略对比是评测平台的职责。策略字段保留在配置里，便于以后扩展
  —— 但**写了没实现的策略会在启动时报错**（见 `rag/config.py` 的 `_validate_config`），
  不做「配了却静默按 fixed_size 跑」的静默失效。
"""

from typing import Dict, List

from rag.logging_utils import get_logger

logger = get_logger()

#: 本切分器支持的策略名（与 rag/config.py 的白名单一致）
SUPPORTED_STRATEGY = "fixed_size"


class FixedSizeChunker:
    """按字符数滑窗切分：窗口 chunk_size，相邻窗口重叠 overlap 个字符。

    为什么要重叠：句子/要点常被切在窗口边界上，重叠让边界信息在前后两个 chunk
    里都完整出现一次，避免「检索到了正确的 chunk 但关键句被切成两半」。
    """

    def __init__(self, config: Dict):
        chunk_cfg = config["chunking"]
        # 双保险：配置是手工拼出来的（绕过 load_config）也不允许「配了 semantic 却按 fixed 跑」
        strategy = str(chunk_cfg.get("strategy", SUPPORTED_STRATEGY)).strip().lower()
        if strategy != SUPPORTED_STRATEGY:
            raise ValueError(
                f"FixedSizeChunker 只支持 chunking.strategy={SUPPORTED_STRATEGY}，当前为 {strategy!r}"
            )
        self.chunk_size = int(chunk_cfg["chunk_size"])
        self.chunk_overlap = int(chunk_cfg["chunk_overlap"])
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")

    def chunk(self, text: str, metadata: Dict | None = None) -> List[Dict]:
        """切分文本，返回 [{text, chunk_index, total_chunks, char_count, **metadata}, ...]"""
        pieces: List[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start : start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap

        chunks = []
        for i, piece in enumerate(pieces):
            stripped = piece.strip()
            if not stripped:  # 纯空白片段没有检索价值，丢弃
                continue
            chunks.append(
                {
                    "text": stripped,
                    "chunk_index": i,  # 文档内序号，建索引时会被替换为全局序号
                    "total_chunks": len(pieces),
                    "char_count": len(stripped),
                    **(metadata or {}),
                }
            )

        logger.info(
            f"切分完成：{len(chunks)} 个 chunk（size={self.chunk_size}, overlap={self.chunk_overlap}）"
        )
        return chunks
