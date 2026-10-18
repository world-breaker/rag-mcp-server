"""端到端验证：用官方 MCP 客户端连接「已在运行的 HTTP 传输」Server。

与 `mcp_client_demo.py`（stdio，客户端拉起子进程）的区别：
- stdio：客户端负责启动 server 子进程，二者通过 stdin/stdout 通信；
- 本脚本：server 已作为独立进程/服务运行，客户端通过 HTTP 连接，
  这是「远程部署、多客户端共享一个知识库服务」的形态。

先启动服务（另开一个终端）：
    .venv\\Scripts\\python.exe server.py --transport streamable-http --port 8000

再运行本脚本：
    .venv\\Scripts\\python.exe scripts\\mcp_http_client_demo.py
    # 或指定地址 / SSE：
    .venv\\Scripts\\python.exe scripts\\mcp_http_client_demo.py --url http://127.0.0.1:8000/mcp
    .venv\\Scripts\\python.exe scripts\\mcp_http_client_demo.py --transport sse --port 8000
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP HTTP 客户端端到端验证")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "sse"],
        default="streamable-http",
        help="HTTP 传输方式（需与服务端一致）",
    )
    parser.add_argument("--url", default=None, help="完整 URL；缺省按 host/port/path 拼")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--path",
        default=None,
        help="HTTP 路径：streamable-http 默认 /mcp，sse 默认 /sse",
    )
    return parser.parse_args(argv)


def _build_url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url
    path = args.path or ("/mcp" if args.transport == "streamable-http" else "/sse")
    return f"http://{args.host}:{args.port}{path}"


async def _run(url: str, transport: str) -> int:
    if transport == "streamable-http":
        from mcp.client.streamable_http import streamable_http_client

        # mcp 2.x：yield (read_stream, write_stream)
        cm = streamable_http_client(url)
    else:
        from mcp.client.sse import sse_client

        cm = sse_client(url)

    print(f"[client] 连接 {transport} → {url}")
    async with cm as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            # 1) 握手
            init_result = await session.initialize()
            server_info = getattr(init_result, "server_info", None) or init_result.serverInfo
            print(f"[client] initialize OK → server={server_info.name} v{server_info.version}")

            # 2) tools/list
            tools = await session.list_tools()
            print(f"[client] tools/list → {len(tools.tools)} 个工具：{[t.name for t in tools.tools]}")

            # 3) tools/call：检索
            for query in ["LoRA 微调是怎么工作的？", "RAG 系统怎么做检索优化？", "明天武汉天气如何？"]:
                result = await session.call_tool(
                    "search_knowledge", {"query": query, "top_k": 2}
                )
                payload = json.loads(result.content[0].text)
                is_error = getattr(result, "is_error", None)
                if is_error is None:
                    is_error = getattr(result, "isError", False)
                print(
                    f"\n[client] tools/call search_knowledge(query={query!r}, top_k=2) "
                    f"isError={is_error} matched={payload['matched']}"
                )
                for r in payload["results"]:
                    preview = r["text"].replace("\n", " ")[:50]
                    print(
                        f"         rank={r['rank']} score={r['score']} "
                        f"source={r['source']} | {preview}..."
                    )
                if payload["matched"] == 0:
                    print(f"         hint: {payload.get('hint')}")

            # 4) tools/call：元信息
            stats_result = await session.call_tool("knowledge_base_stats", {})
            stats = json.loads(stats_result.content[0].text)
            print(
                f"\n[client] tools/call knowledge_base_stats → 文档={stats['documents']} "
                f"chunk={stats['chunks']} 维度={stats['vector_dim']} "
                f"缓存命中={stats['cache_hits']}"
            )

    print(f"\n[OK] MCP {transport} 端到端调用成功（客户端与 server 通过 HTTP 通信）")
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    url = _build_url(args)
    try:
        return asyncio.run(_run(url, args.transport))
    except Exception as exc:
        print(f"\n[FAIL] 连接/调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "       请确认服务已启动，且传输方式与地址一致。例如：\n"
            "       .venv\\Scripts\\python.exe server.py --transport streamable-http --port 8000",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
