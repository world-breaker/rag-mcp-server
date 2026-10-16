"""端到端验证：用官方 MCP 客户端以 stdio 方式拉起 server 子进程，走完整协议调用。

这一步和「Claude Desktop / Cursor 接入」走的是同一条代码路径：
客户端进程 → 启动 server 子进程 → initialize 握手 → tools/list → tools/call。
冒烟测试只验证检索逻辑，本脚本验证协议层，两者互补。

运行：
    .venv\\Scripts\\python.exe scripts\\mcp_client_demo.py
"""

import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER_SCRIPT = REPO_ROOT / "server.py"


async def main() -> int:
    # 关键点：用「当前解释器」拉起 server，保证子进程用的就是本仓库 venv 环境
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(REPO_ROOT),
    )

    print(f"[client] 启动 server 子进程：{sys.executable} {SERVER_SCRIPT.name}")
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) 握手
            init_result = await session.initialize()
            # mcp 2.x 用 snake_case（server_info）；对 1.x 的 camelCase 保持兼容
            server_info = getattr(init_result, "server_info", None) or init_result.serverInfo
            print(f"[client] initialize OK → server={server_info.name} v{server_info.version}")

            # 2) 列出工具（对应 MCP 的 tools/list）
            tools = await session.list_tools()
            print(f"[client] tools/list → {len(tools.tools)} 个工具：{[t.name for t in tools.tools]}")
            for tool in tools.tools:
                print(f"         - {tool.name}: {(tool.description or '')[:70]}...")

            # 3) 调用检索工具（对应 MCP 的 tools/call）
            for query in ["LoRA 微调是怎么工作的？", "RAG 系统怎么做检索优化？", "明天武汉天气如何？"]:
                result = await session.call_tool("search_knowledge", {"query": query, "top_k": 2})
                payload = json.loads(result.content[0].text)
                is_error = getattr(result, "is_error", None)
                if is_error is None:
                    is_error = getattr(result, "isError", False)
                print(f"\n[client] tools/call search_knowledge(query={query!r}, top_k=2)")
                print(f"         isError={is_error} matched={payload['matched']} "
                      f"filtered_by_threshold={payload['filtered_by_threshold']}")
                for r in payload["results"]:
                    preview = r["text"].replace("\n", " ")[:50]
                    print(f"         rank={r['rank']} score={r['score']} source={r['source']} | {preview}...")
                if payload["matched"] == 0:
                    print(f"         hint: {payload.get('hint')}")

            # 4) 调用元信息工具
            stats_result = await session.call_tool("knowledge_base_stats", {})
            stats = json.loads(stats_result.content[0].text)
            print(f"\n[client] tools/call knowledge_base_stats → 文档={stats['documents']} "
                  f"chunk={stats['chunks']} 维度={stats['vector_dim']} 缓存命中={stats['cache_hits']}")

    print("\n[OK] MCP stdio 端到端调用成功（客户端与 server 是两个进程，走 JSON-RPC）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
