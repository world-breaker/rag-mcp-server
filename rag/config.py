"""配置模块：读取 config.yaml，合并默认值，并把相对路径解析成绝对路径。

设计取舍：
- 默认值写在代码里，配置文件只覆盖需要的项 —— 少一项配置不会让服务起不来；
- 所有相对路径按「配置文件所在目录」解析，因此无论在哪个工作目录启动进程，
  行为都一致（MCP 客户端启动 server 时工作目录不可控，这一点很关键）。
"""

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

# 代码内默认值：与 config.yaml 保持一致，缺项时兜底
DEFAULTS: Dict[str, Any] = {
    "knowledge_base": {
        "path": "./data/sample/knowledge_base.json",
    },
    "chunking": {
        "strategy": "fixed_size",
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "embedding": {
        "model_name": "BAAI/bge-small-zh-v1.5",
        "device": "auto",
        "batch_size": 64,
        "cache_dir": "./data/embedding_cache",
        "offline": True,
    },
    "retrieval": {
        "top_k": 3,
        "similarity_threshold": 0.3,
        "max_top_k": 10,
    },
    "server": {
        "name": "rag-mcp-server",
        "version": "0.1.0",
        "warmup": True,
        "log_level": "INFO",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base，未出现的键保留默认值。"""
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """加载配置：默认读仓库根目录的 config.yaml。"""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULTS, user_config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = config_path.parent

    # 相对路径 → 绝对路径
    for section, key in [
        ("knowledge_base", "path"),
        ("embedding", "cache_dir"),
    ]:
        raw = Path(str(config[section][key]))
        if not raw.is_absolute():
            config[section][key] = str((config_path.parent / raw).resolve())
        else:
            config[section][key] = str(raw)

    return config


def resolve_device(config: Dict[str, Any]) -> str:
    """device=auto 时自动探测：有 CUDA 用 GPU，否则 CPU（保证在任何机器都能跑起来）。"""
    device = str(config["embedding"]["device"]).lower()
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch 未装好时不要让探测本身成为故障点
        return "cpu"
