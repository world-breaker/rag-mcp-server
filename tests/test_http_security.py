"""HTTP 安全层测试：Bearer 鉴权 / 令牌桶限流 / `/healthz` / `/metrics`。

分两层：
1. **纯单元层**（毫秒级）：直接驱动 ASGI 中间件、令牌桶、指标表——不启动服务器、不加载模型；
2. **端到端层**（一个测试）：以子进程启动真实 server（开启鉴权 + 极小限流阈值），
   用标准库 `urllib` 打真实 HTTP 端点，并用官方 MCP 客户端带 token 走完整协议。

这样分层的理由：401/429 的**边界**用注入时间的单元测试测（确定性强、失败好定位），
而「中间件真的挂在 uvicorn 上、豁免路径真的豁免」只能靠真实进程验证——
只在单测里过、没在真实链路上过的安全策略，等于没有。

运行：
    .venv\\Scripts\\python.exe -m pytest tests/test_http_security.py -v
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import server  # noqa: E402  （导入只加载配置/创建空服务，不加载模型）
from rag.config import load_config  # noqa: E402
from rag.http_security import (  # noqa: E402
    RateLimiter,
    SecurityMiddleware,
    SecuritySettings,
    TokenBucket,
)
from rag.metrics import Metrics  # noqa: E402


# ==================== 1. 测试脚手架 ====================

class _Captured:
    """收集 ASGI 响应（状态码 / 头 / body），不引入 httpx 之类的额外依赖。"""

    def __init__(self) -> None:
        self.status: int | None = None
        self.headers: dict[str, str] = {}
        self.body: bytes = b""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> dict:
        return json.loads(self.body)


class _Harness:
    """把 SecurityMiddleware 包成可反复调用的驱动器。

    ★ 必须复用同一个 harness：限流桶、指标表的状态是**跨请求**的，
      每个请求都新建中间件会把「第 3 次请求被限流」这类断言变成永远为真/假。
    """

    def __init__(self, settings: SecuritySettings, metrics: Metrics | None = None, inner_status: int = 200):
        self.inner_status = int(inner_status)
        self.inner_calls = 0
        self.middleware = SecurityMiddleware(self._inner, settings=settings, metrics=metrics)

    async def _inner(self, scope, receive, send):
        """被保护的应用：被调用即说明中间件放行了（模拟 MCP 端点）。"""
        self.inner_calls += 1
        await send(
            {
                "type": "http.response.start",
                "status": self.inner_status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"inner": true}'})

    def call(
        self,
        path: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        client: tuple[str, int] = ("203.0.113.7", 51234),
    ) -> _Captured:
        captured = _Captured()
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in (headers or {}).items()
            ],
            "client": client,
            "server": ("127.0.0.1", 8000),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                captured.status = int(message["status"])
                captured.headers = {
                    name.decode("latin-1").lower(): value.decode("latin-1")
                    for name, value in message["headers"]
                }
            elif message["type"] == "http.response.body":
                captured.body += message.get("body", b"")

        asyncio.run(self.middleware(scope, receive, send))
        return captured


def _auth_settings(**overrides) -> SecuritySettings:
    base = dict(auth_enabled=True, token="s3cret", rps=0.0, burst=0)
    base.update(overrides)
    return SecuritySettings(**base)


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
        "auth": {"enabled": False, "token": ""},
        "rate_limit": {"rps": 0, "burst": 0},
    }
    server_cfg.update(server_overrides)
    return {"server": server_cfg}


# ==================== 2. 鉴权 ====================

def test_request_without_token_is_401():
    metrics = Metrics()
    harness = _Harness(_auth_settings(), metrics=metrics)

    response = harness.call("/mcp")

    assert response.status == 401
    assert response.json()["error"] == "unauthorized"
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert harness.inner_calls == 0  # 401 时绝不能触达 MCP 应用
    assert metrics.snapshot()["auth_failures_total"] == 1


def test_request_with_wrong_token_is_401():
    harness = _Harness(_auth_settings())
    response = harness.call("/mcp", headers={"Authorization": "Bearer not-the-token"})

    assert response.status == 401
    assert harness.inner_calls == 0


@pytest.mark.parametrize(
    "header_value",
    ["Bearer s3cret", "bearer s3cret", "BEARER s3cret"],  # scheme 大小写不敏感（RFC 7235）
)
def test_request_with_correct_token_passes(header_value):
    harness = _Harness(_auth_settings())
    response = harness.call("/mcp", headers={"Authorization": header_value})

    assert response.status == 200
    assert harness.inner_calls == 1


@pytest.mark.parametrize("header_value", ["Basic czNjcmV0", "s3cret", "Token s3cret"])
def test_non_bearer_scheme_is_rejected(header_value):
    """`Basic ...` / 裸 token / 别的 scheme 都不算 Bearer 凭证，等同未携带。"""
    harness = _Harness(_auth_settings())
    response = harness.call("/mcp", headers={"Authorization": header_value})

    assert response.status == 401
    assert harness.inner_calls == 0


def test_auth_disabled_lets_requests_through():
    harness = _Harness(SecuritySettings(auth_enabled=False))
    response = harness.call("/mcp")

    assert response.status == 200
    assert harness.inner_calls == 1


def test_auth_enabled_without_token_fails_fast():
    """开了鉴权却没有 token → 构造即报错，避免「以为开了其实没开」这种最危险的配置。"""
    with pytest.raises(ValueError, match="RAG_MCP_AUTH_TOKEN"):
        SecuritySettings(auth_enabled=True, token="")


# ==================== 3. 令牌桶限流（注入时间，完全确定） ====================

def test_token_bucket_allows_burst_then_refills_over_time():
    bucket = TokenBucket(rate=2.0, capacity=3, now=0.0)

    assert [bucket.consume(now=0.0)[0] for _ in range(3)] == [True, True, True]  # burst 起步

    allowed, wait = bucket.consume(now=0.0)
    assert allowed is False
    assert wait == pytest.approx(0.5)  # rate=2/s → 差 1 个令牌 = 0.5s

    assert bucket.consume(now=0.4)[0] is False  # 只补了 0.8 个，还不够
    assert bucket.consume(now=0.5)[0] is True  # 补满 1 个 → 放行


def test_token_bucket_never_exceeds_capacity():
    bucket = TokenBucket(rate=100.0, capacity=2, now=0.0)
    bucket.consume(now=0.0)
    bucket.consume(now=0.0)
    bucket.consume(now=0.0)

    # 空闲 10 秒最多回到 capacity=2，而不是攒出 1000 个令牌（否则「限流」形同虚设）
    assert bucket.consume(now=10.0)[0] is True
    assert bucket.consume(now=10.0)[0] is True
    assert bucket.consume(now=10.0)[0] is False


def test_rate_limiter_is_per_identity():
    limiter = RateLimiter(rps=0.001, burst=1)

    assert limiter.allow(("ip", "1.1.1.1"), now=0.0)[0] is True
    assert limiter.allow(("ip", "1.1.1.1"), now=0.0)[0] is False
    # 另一个身份（另一个 token / 另一个 IP）有独立的桶，不被别人拖累
    assert limiter.allow(("token", "other"), now=0.0)[0] is True
    assert limiter.allow(("ip", "2.2.2.2"), now=0.0)[0] is True


def test_rate_limiter_disabled_when_rps_is_zero():
    limiter = RateLimiter(rps=0.0, burst=0)

    assert limiter.enabled is False
    assert all(limiter.allow(("ip", "1.1.1.1"), now=0.0)[0] for _ in range(100))


def test_rate_limiter_evicts_idle_buckets_to_bound_memory():
    limiter = RateLimiter(rps=1.0, burst=1, max_buckets=10)
    for i in range(50):
        limiter.allow(("ip", f"10.0.0.{i}"), now=float(i))

    assert len(limiter._buckets) <= 10  # 内存有上限，不会被随机 IP 打爆


def test_burst_defaults_to_one_second_of_quota():
    assert SecuritySettings(auth_enabled=False, rps=2.5, burst=0).burst_capacity == 3
    assert SecuritySettings(auth_enabled=False, rps=10, burst=40).burst_capacity == 40


# ==================== 4. 429 + Retry-After ====================

def test_third_request_is_429_with_retry_after():
    metrics = Metrics()
    harness = _Harness(_auth_settings(token="t", rps=0.5, burst=2), metrics=metrics)
    headers = {"Authorization": "Bearer t"}

    assert harness.call("/mcp", headers=headers).status == 200
    assert harness.call("/mcp", headers=headers).status == 200

    response = harness.call("/mcp", headers=headers)
    assert response.status == 429
    assert harness.inner_calls == 2  # 被限流的请求没有进到 MCP 应用
    assert response.json()["error"] == "rate_limited"
    retry_after = response.headers["retry-after"]
    assert retry_after.isdigit() and int(retry_after) >= 1
    assert int(retry_after) == 2  # rate=0.5/s、桶已空 → 2 秒才补回 1 个令牌
    assert metrics.snapshot()["rate_limited_total"] == 1


def test_rate_limit_applies_before_auth_check():
    """限流在鉴权之前生效：匿名洪水同样被计价，挡住「拿错误 token 暴力猜」。"""
    harness = _Harness(_auth_settings(token="t", rps=0.001, burst=1))

    assert harness.call("/mcp").status == 401
    assert harness.call("/mcp").status == 429  # 第 2 发匿名请求已超限，而不是继续 401
    assert harness.inner_calls == 0


def test_healthz_and_metrics_are_exempt_from_auth_and_rate_limit():
    harness = _Harness(_auth_settings(token="t", rps=0.001, burst=1))

    for path in ("/healthz", "/metrics"):
        for _ in range(20):  # 无凭证 + 远超限流阈值，也必须全部放行
            assert harness.call(path, method="GET").status == 200, path


def test_exempt_path_trailing_slash_is_tolerated():
    harness = _Harness(_auth_settings())
    assert harness.call("/healthz/", method="GET").status == 200


# ==================== 5. 指标（/metrics 的内容） ====================

def test_metrics_render_contains_all_required_metrics():
    metrics = Metrics()
    metrics.record_tool_call("search_knowledge")
    metrics.record_tool_call("search_knowledge")
    metrics.record_tool_call("knowledge_base_stats")
    metrics.record_request(200, 12.0)
    metrics.record_request(429, 1.0)
    metrics.record_request(500, 30.0)  # 5xx 同时计入 errors_total
    metrics.record_error()  # 工具内部异常也计入同一个 errors_total

    text = metrics.render_prometheus()

    assert "# TYPE rag_mcp_requests_total counter" in text
    assert 'rag_mcp_requests_total{status="200"} 1' in text
    assert 'rag_mcp_requests_total{status="429"} 1' in text
    assert 'rag_mcp_tool_calls_total{tool="search_knowledge"} 2' in text
    assert 'rag_mcp_tool_calls_total{tool="knowledge_base_stats"} 1' in text
    assert "rag_mcp_errors_total 2" in text
    assert 'rag_mcp_request_duration_ms{quantile="0.5"}' in text
    assert 'rag_mcp_request_duration_ms{quantile="0.95"}' in text
    assert 'rag_mcp_request_duration_ms{quantile="0.99"}' in text
    assert "rag_mcp_request_duration_ms_count 3" in text
    assert "rag_mcp_request_duration_ms_sum 43.0" in text
    assert "rag_mcp_uptime_seconds" in text


def test_metrics_summary_reports_percentiles_and_bounds_memory():
    metrics = Metrics(max_samples=100)
    for i in range(1, 101):
        metrics.record_request(200, float(i))  # 1..100 ms

    snapshot = metrics.snapshot()["request_duration_ms"]
    assert snapshot["count"] == 100
    assert snapshot["p50"] == 50.0
    assert snapshot["p95"] == 95.0
    assert snapshot["p99"] == 99.0

    for _ in range(1000):  # deque 有上限：内存不随请求数无限增长
        metrics.record_request(200, 1.0)
    assert metrics.snapshot()["request_duration_ms"]["count"] == 1100  # 计数是全量
    assert len(metrics._durations) == 100  # 分位数只用最近 100 个采样


def test_middleware_records_status_and_error_metrics():
    metrics = Metrics()
    harness = _Harness(SecuritySettings(), metrics=metrics)
    harness.call("/mcp")
    harness.inner_status = 500
    harness.call("/mcp")

    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == {200: 1, 500: 1}  # 状态码为 int（JSON 序列化时转字符串）
    assert snapshot["errors_total"] == 1
    assert snapshot["request_duration_ms"]["count"] == 2


def test_middleware_records_500_when_inner_app_raises():
    """内部应用抛异常时也要记一条 5xx 指标，然后原样抛出（不能吞掉协议层错误）。"""
    metrics = Metrics()

    async def boom(scope, receive, send):
        raise RuntimeError("inner boom")

    middleware = SecurityMiddleware(boom, settings=SecuritySettings(), metrics=metrics)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [],
        "client": ("1.2.3.4", 1),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):  # pragma: no cover
        pass

    with pytest.raises(RuntimeError):
        asyncio.run(middleware(scope, receive, send))

    assert metrics.snapshot()["errors_total"] == 1


def test_lifespan_scope_passes_through_untouched():
    """uvicorn 的 lifespan 事件不是 http 请求，必须原样透传（否则会话管理器起不来）。"""
    seen = {}

    async def app(scope, receive, send):
        seen["scope"] = scope

    async def receive():
        return {"type": "lifespan.startup"}

    async def send(message):  # pragma: no cover
        pass

    middleware = SecurityMiddleware(app, settings=_auth_settings(), metrics=Metrics())
    lifespan_scope = {"type": "lifespan"}
    asyncio.run(middleware(lifespan_scope, receive, send))

    assert seen["scope"] is lifespan_scope


# ==================== 6. 配置层：默认关闭 + 环境变量优先 ====================

def test_default_config_ships_security_off():
    """默认必须是「stdio 且不开鉴权、不限流」——这是既有 14 个测试零回归的前提。"""
    config = load_config()

    assert config["server"]["transport"] == "stdio"
    assert config["server"]["host"] == "127.0.0.1"
    assert config["server"]["auth"]["enabled"] is False
    assert config["server"]["rate_limit"]["rps"] == 0
    assert config["server"]["security_exempt_paths"] == ["/healthz", "/metrics"]


def test_env_token_overrides_config_and_enables_auth(monkeypatch):
    monkeypatch.setenv("RAG_MCP_AUTH_TOKEN", "env-token-123")
    config = load_config()

    assert config["server"]["auth"]["token"] == "env-token-123"
    assert config["server"]["auth"]["enabled"] is True  # 设了 token 就不允许「忘了开 enabled」裸奔
    # 环境变量注入的配置也要能正常走到传输参数解析（HTTP 才算数，stdio 不受影响）
    transport, kwargs = server.resolve_transport(_args(transport="streamable-http"), config)
    assert transport == "streamable-http"
    assert kwargs["port"] == 8000


def test_env_rate_limit_overrides_config(monkeypatch):
    monkeypatch.setenv("RAG_MCP_RATE_LIMIT_RPS", "0.5")
    monkeypatch.setenv("RAG_MCP_RATE_LIMIT_BURST", "7")
    config = load_config()

    assert config["server"]["rate_limit"]["rps"] == 0.5
    assert config["server"]["rate_limit"]["burst"] == 7


def test_stdio_transport_is_unaffected_by_security_settings():
    """硬约束：鉴权/限流只作用于 HTTP 传输；stdio 分支连 kwargs 都不带安全参数。"""
    config = _cfg(auth={"enabled": True, "token": "t"}, rate_limit={"rps": 1, "burst": 1})

    transport, kwargs = server.resolve_transport(_args(transport="stdio"), config)

    assert transport == "stdio"
    assert kwargs == {}


# ==================== 7. 端到端：真实进程 + 真实 HTTP 端点 ====================

E2E_TOKEN = "e2e-token-please-change"
E2E_RPS = "0.05"  # 20 秒才补 1 个令牌：连打必然打爆，且不受机器快慢影响
E2E_BURST = "10"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


#: 不走系统代理：本机全局代理（http.proxy=127.0.0.1:7897）会把 localhost 请求也代理走 → 502
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
):
    """返回 (status, headers, text)；HTTP 错误码不抛异常，交给断言处理。

    headers 的键统一转小写，避免 `Content-Type` / `content-type` 的大小写差异影响断言。
    """
    request = urllib.request.Request(url, method=method, headers=headers or {}, data=body)
    try:
        with _OPENER.open(request, timeout=10) as response:
            return (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                response.read().decode("utf-8"),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            {k.lower(): v for k, v in exc.headers.items()},
            exc.read().decode("utf-8"),
        )


def _parse_metrics(text: str) -> dict:
    """把 Prometheus 文本解析成 {样本名: 值}，便于断言具体数值。"""
    parsed = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            parsed[name.strip()] = float(value)
        except ValueError:  # pragma: no cover
            continue
    return parsed


async def _call_with_token(url: str, token: str) -> dict:
    """官方 MCP 客户端带 Bearer token 走完整协议：initialize → tools/list → tools/call。"""
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    # mcp 2.x 的 streamable_http_client 不直接收 headers，而是接收一个自定义 http 客户端
    async with httpx2.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}, trust_env=False, timeout=30.0
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                result = await session.call_tool(
                    "search_knowledge", {"query": "LoRA 微调的原理", "top_k": 2}
                )
                return {
                    "tools": [t.name for t in tools.tools],
                    "payload": json.loads(result.content[0].text),
                }


def test_http_security_end_to_end(tmp_path):
    """真实 server（开鉴权 + 极小限流）上的 401 / 429 / healthz / metrics 全链路验证。"""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    stderr_path = tmp_path / "server_stderr.log"
    rpc_body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()

    env = dict(os.environ)
    env.update(
        {
            "RAG_MCP_AUTH_TOKEN": E2E_TOKEN,
            "RAG_MCP_RATE_LIMIT_RPS": E2E_RPS,
            "RAG_MCP_RATE_LIMIT_BURST": E2E_BURST,
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )

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
            env=env,
        )
        try:
            if not _wait_port(port):
                stderr_file.flush()
                pytest.fail(
                    "server 未在超时内监听 127.0.0.1:%d。stderr 末尾：\n%s"
                    % (port, stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:])
                )

            # --- (1) /healthz：无凭证可访问、只读、不触发建索引 ---
            status, _, text = _http(f"{base}/healthz")
            assert status == 200
            health = json.loads(text)
            assert health["status"] == "ok"
            assert health["index_ready"] is True
            assert health["documents"] == 5
            assert health["chunks"] >= 5

            # --- (2) /metrics：无凭证可访问，Prometheus 文本，四类指标齐全 ---
            status, headers, text = _http(f"{base}/metrics")
            assert status == 200
            assert "text/plain" in headers["content-type"]
            for metric in (
                "rag_mcp_requests_total",
                "rag_mcp_tool_calls_total",
                "rag_mcp_errors_total",
                "rag_mcp_request_duration_ms",
                "rag_mcp_rate_limited_total",
                "rag_mcp_auth_failures_total",
            ):
                assert metric in text, metric

            # --- (3) 无 token / 错 token → 401 JSON ---
            status, headers, text = _http(
                f"{base}/mcp",
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                body=rpc_body,
            )
            assert status == 401
            assert json.loads(text)["error"] == "unauthorized"
            assert headers["www-authenticate"].startswith("Bearer")

            status, _, _ = _http(
                f"{base}/mcp",
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer wrong-token"},
                body=rpc_body,
            )
            assert status == 401

            # --- (4) 正确 token：官方 MCP 客户端走完整协议 ---
            outcome = asyncio.run(_call_with_token(f"{base}/mcp", E2E_TOKEN))
            assert {"search_knowledge", "knowledge_base_stats"} <= set(outcome["tools"])
            assert outcome["payload"]["matched"] >= 1
            assert outcome["payload"]["results"][0]["rank"] == 1

            # --- (5) 服务端指标确实记下了这次工具调用（客户端 vs 服务端交叉验证）---
            parsed = _parse_metrics(_http(f"{base}/metrics")[2])
            assert parsed['rag_mcp_tool_calls_total{tool="search_knowledge"}'] >= 1
            assert parsed["rag_mcp_auth_failures_total"] >= 2
            assert parsed['rag_mcp_requests_total{status="200"}'] >= 1
            assert parsed["rag_mcp_request_duration_ms_count"] >= 1

            # --- (6) 健康检查 / 指标不受限流影响（连打 15 轮必须全 200）---
            for _ in range(15):
                assert _http(f"{base}/healthz")[0] == 200
                assert _http(f"{base}/metrics")[0] == 200

            # --- (7) 匿名洪水：burst=10 的桶被打爆 → 429 + Retry-After ---
            statuses = []
            for _ in range(14):
                status, headers, text = _http(
                    f"{base}/mcp",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body=rpc_body,
                )
                statuses.append(status)
                if status == 429:
                    assert json.loads(text)["error"] == "rate_limited"
                    assert int(headers["retry-after"]) >= 1

            assert all(s in (401, 429) for s in statuses), statuses
            assert statuses.count(429) >= 1, f"未撞到限流：{statuses}"
            assert statuses[-1] == 429, f"桶空后仍被放行：{statuses}"

            # 被限流的次数也要能在 /metrics 上看到（可观测性闭环）
            parsed = _parse_metrics(_http(f"{base}/metrics")[2])
            assert parsed["rag_mcp_rate_limited_total"] >= 1
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                proc.kill()
