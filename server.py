"""MCP Server 入口：把 RAG 检索能力暴露成标准 MCP 工具（stdio 传输）。

启动方式（由 MCP 客户端负责拉起，一般不需要手工执行）：
    python server.py

协议要点：
- stdio 模式下 **stdout 是 JSON-RPC 报文通道**，任何 print 都会污染协议，
  因此本文件所有日志都经由 logging_utils 写到 stderr；
- 工具返回值统一是 JSON 字符串（MCP 工具的返回值最终作为文本回到模型上下文，
  结构化 JSON 便于上层程序解析、也便于做工具调用评测）。
"""

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


def main() -> None:
    logger.info(f"MCP Server 启动：{config['server']['name']} v{config['server']['version']}（stdio）")
    logger.info(f"知识库：{config['knowledge_base']['path']}")
    if config["server"].get("warmup", True):
        service.warmup()  # 预热：避免首次工具调用撞上客户端超时
    server.run("stdio")


if __name__ == "__main__":
    main()
    sys.exit(0)
