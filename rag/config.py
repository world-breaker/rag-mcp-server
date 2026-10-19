"""配置模块：读取 config.yaml，合并默认值，并把相对路径解析成绝对路径。

设计取舍：
- 默认值写在代码里，配置文件只覆盖需要的项 —— 少一项配置不会让服务起不来；
- 所有相对路径按「配置文件所在目录」解析，因此无论在哪个工作目录启动进程，
  行为都一致（MCP 客户端启动 server 时工作目录不可控，这一点很关键）。
"""

import os
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
        # 传输方式：stdio（默认）| streamable-http | sse
        "transport": "stdio",
        "host": "127.0.0.1",
        "port": 8000,
        "streamable_http_path": "/mcp",
        "sse_path": "/sse",
        "message_path": "/messages/",
        "stateless_http": False,
        "json_response": False,
        "allowed_hosts": [],
        "allowed_origins": [],
        # 安全层（仅 HTTP 传输生效，stdio 是父子进程模式，不受影响）
        # auth：Bearer 鉴权；enabled=false 时 token 留空即可
        "auth": {
            "enabled": False,
            "token": "",  # 也可用环境变量 RAG_MCP_AUTH_TOKEN（优先级更高，避免密钥进版本库）
        },
        # rate_limit：令牌桶限流；rps<=0 表示关闭（默认关闭：本机/单客户端开发时不希望被自己的探针打到 429）
        "rate_limit": {
            "rps": 0,
            "burst": 0,  # <=0 时取 ceil(rps)（即 1 秒的配额）作为突发量
        },
        # 不鉴权、不限流的路径（健康检查 / 指标采集必须无凭证可访问）
        "security_exempt_paths": ["/healthz", "/metrics"],
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

    # 环境变量覆盖（容器 / CI / 临时切换传输方式时不必改配置文件）
    # 优先级：命令行参数 > 环境变量 > config.yaml > 代码默认值
    for key, env_name in (
        ("transport", "RAG_MCP_TRANSPORT"),
        ("host", "RAG_MCP_HOST"),
        ("port", "RAG_MCP_PORT"),
    ):
        raw = os.environ.get(env_name)
        if raw:
            config["server"][key] = int(raw) if key == "port" else raw

    # 安全层：密钥与阈值几乎总是由环境变量注入（容器编排里 config 文件是只读的、
    # 而且 token 不该进版本库），所以这里也支持环境变量覆盖
    auth_env_token = os.environ.get("RAG_MCP_AUTH_TOKEN")
    if auth_env_token:
        config["server"]["auth"]["token"] = auth_env_token
        # 设置了 token 就默认启用鉴权：避免「配了 token 却因为忘了改 enabled 而裸奔」这个高频坑
        config["server"]["auth"]["enabled"] = True

    raw_rps = os.environ.get("RAG_MCP_RATE_LIMIT_RPS")
    if raw_rps:
        config["server"]["rate_limit"]["rps"] = float(raw_rps)
    raw_burst = os.environ.get("RAG_MCP_RATE_LIMIT_BURST")
    if raw_burst:
        config["server"]["rate_limit"]["burst"] = int(raw_burst)

    # embedding.offline：容器里通常没有 HuggingFace 缓存，需要放开联网下载一次；
    # 生产建议挂载缓存目录并保持 offline=true（见 README Docker 一节）
    raw_offline = os.environ.get("RAG_MCP_EMBEDDING_OFFLINE")
    if raw_offline is not None and raw_offline != "":
        config["embedding"]["offline"] = str(raw_offline).strip().lower() in ("1", "true", "yes", "on")

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
