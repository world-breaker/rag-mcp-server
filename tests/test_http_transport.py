"""HTTP 传输测试。

分两层：
1. `resolve_transport` 纯函数单测——不启动服务器、不加载模型，秒级完成；
2. Streamable HTTP 端到端——以子进程启动真实 server，用官方 MCP 客户端走
   initialize → tools/list → tools/call 完整协议，验证「换传输不换业务逻辑」。

运行：
    .venv\\Scripts\\python.exe -m pytest tests/test_http_transport.py -v
"""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import server  # noqa: E402  （导入只加载配置/创建空服务，不加载模型）


# ========== 1. 参数解析单测 ==========

def _args(**overrides) -> argparse.Namespace:
    base = dict(
        transport=None,
        host=None,
        port=None,
        path=None,
        stateless=False,
        json_response=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _cfg(**server_overrides) -> dict:
    server_cfg = {
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
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg}


def test_default_is_stdio_with_no_kwargs():
    transport, kwargs = server.resolve_transport(_args(), _cfg())
    assert transport == "stdio"
    assert kwargs == {}


def test_streamable_http_maps_config():
    transport, kwargs = server.resolve_transport(
        _args(), _cfg(transport="streamable-http", port=9000)
    )
    assert transport == "streamable-http"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 9000
    assert kwargs["streamable_http_path"] == "/mcp"
    assert kwargs["stateless_http"] is False
    assert kwargs["json_response"] is False


def test_cli_overrides_config():
    transport, kwargs = server.resolve_transport(
        _args(transport="streamable-http", host="0.0.0.0", port=9100, path="/rag/mcp"),
        _cfg(transport="stdio", port=8000),
    )
    assert transport == "streamable-http"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 9100
    assert kwargs["streamable_http_path"] == "/rag/mcp"


def test_stateless_and_json_response_flags():
    _, kwargs = server.resolve_transport(
        _args(transport="streamable-http", stateless=True, json_response=True),
        _cfg(),
    )
    assert kwargs["stateless_http"] is True
    assert kwargs["json_response"] is True


def test_sse_maps_paths():
    transport, kwargs = server.resolve_transport(
        _args(transport="sse", port=9200), _cfg()
    )
    assert transport == "sse"
    assert kwargs["sse_path"] == "/sse"
    assert kwargs["message_path"] == "/messages/"
    assert kwargs["port"] == 9200


def test_explicit_dns_rebinding_allowlist():
    _, kwargs = server.resolve_transport(
        _args(transport="streamable-http"),
        _cfg(allowed_hosts=["127.0.0.1:*", "rag.example.com"],
             allowed_origins=["https://rag.example.com"]),
    )
    security = kwargs["transport_security"]
    assert security.enable_dns_rebinding_protection is True
    assert "rag.example.com" in security.allowed_hosts


def test_unknown_transport_raises():
    with pytest.raises(ValueError):
        server.resolve_transport(_args(transport="websocket"), _cfg())


# ========== 2. Streamable HTTP 端到端 ==========

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, timeout: float = 90.0) -> bool:
    """等待端口可连接（server 会先预热索引，之后 uvicorn 才监听）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


async def _exercise_http_client(port: int) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = f"http://127.0.0.1:{port}/mcp"
    async with streamable_http_client(url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {"search_knowledge", "knowledge_base_stats"} <= names

            stats_result = await session.call_tool("knowledge_base_stats", {})
            stats = json.loads(stats_result.content[0].text)
            assert stats["documents"] == 5
            assert stats["vector_dim"] == 512

            search_result = await session.call_tool(
                "search_knowledge", {"query": "LoRA 微调的原理", "top_k": 2}
            )
            payload = json.loads(search_result.content[0].text)
            assert payload["matched"] >= 1
            assert payload["results"][0]["rank"] == 1


def test_streamable_http_end_to_end(tmp_path):
    port = _free_port()
    stderr_path = tmp_path / "server_stderr.log"

    with open(stderr_path, "w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "server.py"),
                "--transport",
                "streamable-http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        try:
            if not _wait_port(port):
                stderr_file.flush()
                log_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                pytest.fail(
                    f"server 未在超时内监听 127.0.0.1:{port}。stderr 末尾：\n{log_tail}"
                )
            asyncio.run(_exercise_http_client(port))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
