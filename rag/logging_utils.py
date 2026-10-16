"""日志工具。

★ 关键约束：MCP 的 stdio 传输用 **stdout** 传 JSON-RPC 报文，
所以 server 进程里一行 print / 一条 stdout 日志都会破坏协议。
本模块把日志统一写到 **stderr**，从源头避免这个问题。
"""

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "rag_mcp", level: str = "INFO") -> logging.Logger:
    """返回写往 stderr 的 logger（全局只配置一次）。"""
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root = logging.getLogger("rag_mcp")
        root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        root.addHandler(handler)
        root.propagate = False  # 不再向上冒泡，避免被别的 handler 写到 stdout
        _CONFIGURED = True

    if level:
        logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    return logger
