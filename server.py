"""MCP Server 入口：把 RAG 检索能力暴露成标准 MCP 工具。

支持三种传输（业务检索代码完全不变，只换传输层）：
- stdio（默认）：由 MCP 客户端拉起子进程，适合 Claude Desktop / Cursor 本地接入；
- streamable-http（推荐远程）：`python server.py --transport streamable-http --port 8000`，
  端点为 `http://<host>:<port>/mcp`，支持会话与流式；
- sse（旧版 HTTP）：`python server.py --transport sse --port 8000`，端点为 `/sse` + `/messages/`，
  仅为兼容旧客户端保留。

启动方式：
    python server.py                                   # stdio（默认）
    python server.py --transport streamable-http        # HTTP（默认 127.0.0.1:8000/mcp）
    RAG_MCP_TRANSPORT=streamable-http python server.py  # 环境变量方式（容器常用）

协议要点：
- stdio 模式下 **stdout 是 JSON-RPC 报文通道**，任何 print 都会污染协议，
  因此本文件所有日志都经由 logging_utils 写到 stderr；
- HTTP 模式下日志同样走 stderr，不进入响应体；
- 工具返回值统一是 JSON 字符串（MCP 工具的返回值最终作为文本回到模型上下文，
  结构化 JSON 便于上层程序解析、也便于做工具调用评测）。
"""

import argparse
import json
import sys

from mcp.server.mcpserver import MCPServer

from rag.config import load_config
from rag.logging_utils import get_logger
from rag.service import KnowledgeService

config = load_config()
logger = get_logger("rag_mcp.server", level=config["server"].get("log_level", "INFO"))

# 检索服务：进程级单例，索引只构建一次
service = KnowledgeService(config)

server = MCPServer(
    name=config["server"]["name"],
    version=config["server"]["version"],
    instructions=(
        "RAG 知识库检索服务。用 search_knowledge 检索本地知识库并拿到带相似度分数的原文片段；"
        "回答用户问题时先检索、再基于返回片段作答，不要凭记忆编造。"
        "若结果为空，说明知识库中没有相关内容，应如实告知用户。"
    ),
)


@server.tool(
    description=(
        "检索本地知识库（中文技术文档），返回最相关的若干原文片段与相似度分数。"
        "适用场景：用户问题可能依赖私有/本地资料时，先调用本工具拿证据再回答。"
        "参数 query 用自然语言提问；top_k 为返回条数（默认 3，上限 10）。"
        "返回 JSON：results 内含 rank/score/source/chunk_index/text，"
        "matched=0 表示知识库中没有高于相似度阈值的内容。"
    )
)
def search_knowledge(query: str, top_k: int = 3) -> str:
    """检索知识库，返回片段 + 相似度分数（JSON 字符串）。"""
    try:
        payload = service.search(query, top_k=top_k)
    except Exception as exc:  # 把内部异常转成结构化错误，而不是让协议层报错
        logger.exception(f"search_knowledge 失败：{exc}")
        payload = {
            "query": query,
            "error": f"{type(exc).__name__}: {exc}",
            "results": [],
        }
    # ensure_ascii=False：中文原文直接可读，避免 \\u4e2d 转义串占满上下文
    return json.dumps(payload, ensure_ascii=False)


@server.tool(
    description=(
        "返回知识库索引的元信息（文档数、chunk 数、来源文件列表、embedding 模型与维度、"
        "相似度阈值、缓存命中次数等）。用于先判断知识库覆盖范围，再决定是否检索。"
    )
)
def knowledge_base_stats() -> str:
    """返回知识库索引元信息（JSON 字符串）。"""
    try:
        return json.dumps(service.stats(), ensure_ascii=False)
    except Exception as exc:
        logger.exception(f"knowledge_base_stats 失败：{exc}")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def _parse_args(argv=None) -> argparse.Namespace:
    """命令行参数：优先级高于 config.yaml 与环境变量。"""
    parser = argparse.ArgumentParser(
        description="rag-mcp-server：把 RAG 知识库检索暴露成标准 MCP 工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default=None,
        help="传输方式；缺省读 config.yaml 的 server.transport（默认 stdio）",
    )
    parser.add_argument("--host", default=None, help="HTTP 监听地址（仅 HTTP 传输生效）")
    parser.add_argument("--port", type=int, default=None, help="HTTP 监听端口（仅 HTTP 传输生效）")
    parser.add_argument(
        "--path",
        default=None,
        help="HTTP 路径：streamable-http 默认 /mcp，sse 默认 /sse",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="streamable-http 无状态模式（不保存会话，适合多实例/负载均衡部署）",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help="streamable-http 返回普通 JSON 而非 SSE 流（便于 curl / httpx 调试）",
    )
    return parser.parse_args(argv)


def _transport_security(server_cfg: dict):
    """按配置构造 DNS rebinding 防护；未配置则返回 None，交给 SDK 默认策略。

    SDK 默认：仅当 host 为 127.0.0.1 / localhost / ::1 时自动开启防护并放行本机 Host；
    绑定 0.0.0.0 时默认不限制。对外暴露到域名/IP 时建议显式配置 allowed_hosts/allowed_origins。
    """
    allowed_hosts = list(server_cfg.get("allowed_hosts") or [])
    allowed_origins = list(server_cfg.get("allowed_origins") or [])
    if not allowed_hosts and not allowed_origins:
        return None

    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def resolve_transport(args: argparse.Namespace, cfg: dict) -> tuple[str, dict]:
    """把「命令行 > 环境变量(已并入 cfg) > config.yaml」解析成 (transport, run_kwargs)。

    抽成纯函数便于单测：不启动服务器、不加载模型，只验证参数映射是否正确。
    """
    server_cfg = cfg["server"]
    transport = args.transport or server_cfg.get("transport", "stdio")
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = int(args.port if args.port is not None else server_cfg.get("port", 8000))

    if transport == "stdio":
        return "stdio", {}

    security = _transport_security(server_cfg)

    if transport == "streamable-http":
        kwargs = {
            "host": host,
            "port": port,
            "streamable_http_path": args.path
            or server_cfg.get("streamable_http_path", "/mcp"),
            "stateless_http": bool(
                server_cfg.get("stateless_http", False) or args.stateless
            ),
            "json_response": bool(
                server_cfg.get("json_response", False) or args.json_response
            ),
        }
        if security is not None:
            kwargs["transport_security"] = security
        return transport, kwargs

    if transport == "sse":
        kwargs = {
            "host": host,
            "port": port,
            "sse_path": args.path or server_cfg.get("sse_path", "/sse"),
            "message_path": server_cfg.get("message_path", "/messages/"),
        }
        if security is not None:
            kwargs["transport_security"] = security
        return transport, kwargs

    raise ValueError(
        f"未知传输方式：{transport!r}（可选 stdio / streamable-http / sse）"
    )


def _endpoint_hint(transport: str, kwargs: dict) -> str:
    if transport == "streamable-http":
        return f"http://{kwargs['host']}:{kwargs['port']}{kwargs['streamable_http_path']}"
    if transport == "sse":
        return f"http://{kwargs['host']}:{kwargs['port']}{kwargs['sse_path']}"
    return "stdin/stdout"


def main(argv=None) -> None:
    args = _parse_args(argv)
    transport, kwargs = resolve_transport(args, config)

    logger.info(
        f"MCP Server 启动：{config['server']['name']} v{config['server']['version']}"
        f"（{transport}）"
    )
    logger.info(f"知识库：{config['knowledge_base']['path']}")
    logger.info(f"端点：{_endpoint_hint(transport, kwargs)}")
    if transport == "streamable-http" and kwargs.get("stateless_http"):
        logger.info("streamable-http 运行在 stateless 模式（不保存会话状态）")

    if config["server"].get("warmup", True):
        service.warmup()  # 预热：避免首次工具调用撞上客户端超时

    server.run(transport, **kwargs)


if __name__ == "__main__":
    main()
    sys.exit(0)
