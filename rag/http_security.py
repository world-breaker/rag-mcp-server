"""HTTP 传输的安全层：Bearer 鉴权 + 令牌桶限流（纯 ASGI 中间件，零额外依赖）。

**只作用于 HTTP 传输**：stdio 模式下客户端与 Server 是同一台机器上的父子进程，
不存在网络入口，鉴权/限流没有意义（README §10 有说明）。

为什么用「自己写的纯 ASGI 中间件」而不是 SDK 自带的 `token_verifier`：
- SDK 的 `auth` 走的是 **MCP OAuth 2.1** 那一套（授权服务器元数据、动态客户端注册、
  `WWW-Authenticate` 里带 resource metadata URL），面向的是「多个第三方客户端接入一个远程服务」；
  本项目是**单知识库 + 自己的客户端/Agent**，一个共享 token 就够，复杂度和收益不成比例；
- 纯 ASGI 中间件不依赖 Starlette 的路由/请求对象，可以脱离框架直接单测（见 tests/test_http_security.py），
  也不会碰请求体（不会破坏 streamable-http 的 SSE 流式响应）。

职责与顺序（一次请求的处理链）：

    /healthz、/metrics  → 直接放行（只读探针，不鉴权不限流）→ 但要计入指标
    其他路径 →
        1) 取身份：`Authorization: Bearer <token>` 有则按 token 计价，无则按客户端 IP
        2) 令牌桶限流：超限 → 429 + Retry-After
        3) Bearer 校验（auth.enabled=true 时）：失败 → 401 + WWW-Authenticate
        4) 放行到 MCP 应用，记录状态码与耗时

为什么限流在鉴权**之前**：未携带正确 token 的洪水请求同样会占用 CPU（协议解析、TLS、日志），
先限流才能挡住「拿错误 token 暴力猜」；而按 token/IP 分桶保证合法客户端不会被别人拖累。
"""

import hmac
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

from rag.logging_utils import get_logger

logger = get_logger("rag_mcp.security")

#: 默认豁免路径：健康检查与指标必须能在「没有凭证」的情况下被 k8s / Prometheus 采集
DEFAULT_EXEMPT_PATHS: Tuple[str, ...] = ("/healthz", "/metrics")

#: 限流表最大键数（每个键一个令牌桶）；超出时优先淘汰「已回满」的桶（淘汰无状态损失）
DEFAULT_MAX_BUCKETS = 4096


@dataclass(frozen=True)
class SecuritySettings:
    """安全配置（来自 config.yaml 的 server.auth / server.rate_limit，环境变量已并入）。"""

    auth_enabled: bool = False
    token: str = ""
    rps: float = 0.0
    burst: int = 0
    exempt_paths: Tuple[str, ...] = DEFAULT_EXEMPT_PATHS

    def __post_init__(self) -> None:
        # 开了鉴权却没有 token → 启动即失败（fail fast），绝不「静默放行」
        if self.auth_enabled and not self.token:
            raise ValueError(
                "server.auth.enabled=true 但没有配置 token："
                "请在 config.yaml 填 server.auth.token，或设置环境变量 RAG_MCP_AUTH_TOKEN"
            )

    @property
    def rate_limit_enabled(self) -> bool:
        return self.rps > 0

    @property
    def burst_capacity(self) -> int:
        """令牌桶容量：burst 未配置（<=0）时取 max(1, ceil(rps))，即「1 秒的配额」作为突发量。"""
        if self.burst and self.burst > 0:
            return int(self.burst)
        return max(1, int(math.ceil(self.rps))) if self.rps > 0 else 0

    @classmethod
    def from_config(cls, server_cfg: MutableMapping[str, Any]) -> "SecuritySettings":
        """从 server 配置段构造（缺项用默认值，与 rag/config.py 的兜底策略一致）。"""
        auth = dict(server_cfg.get("auth") or {})
        rate_limit = dict(server_cfg.get("rate_limit") or {})
        # 环境变量 RAG_MCP_AUTH_TOKEN 已在 config.py 里并入 auth.token，并用它隐式开启鉴权
        return cls(
            auth_enabled=bool(auth.get("enabled", False)),
            token=str(auth.get("token") or ""),
            rps=float(rate_limit.get("rps", 0) or 0),
            burst=int(rate_limit.get("burst", 0) or 0),
            exempt_paths=tuple(server_cfg.get("security_exempt_paths") or DEFAULT_EXEMPT_PATHS),
        )

    def describe(self) -> str:
        """启动日志用的一行描述（不打印 token 本身）。"""
        parts: List[str] = []
        parts.append("Bearer 鉴权=开启" if self.auth_enabled else "Bearer 鉴权=关闭")
        if self.rate_limit_enabled:
            parts.append(f"限流={self.rps:g} rps / burst {self.burst_capacity}")
        else:
            parts.append("限流=关闭")
        parts.append(f"豁免路径={','.join(self.exempt_paths)}")
        return "；".join(parts)


class TokenBucket:
    """令牌桶：容量 `capacity`，按 `rate`（个/秒）连续补充。

    惰性补充（lazy refill）：只在被访问时按时间差补令牌，因此空桶客户端不占任何后台线程；
    这与「固定窗口计数」的区别是允许突发（最多 capacity 个）但平均速率被 `rate` 限住。
    """

    __slots__ = ("capacity", "rate", "tokens", "updated_at")

    def __init__(self, rate: float, capacity: int, now: Optional[float] = None):
        self.capacity = int(capacity)
        self.rate = float(rate)
        self.tokens = float(capacity)
        self.updated_at = time.monotonic() if now is None else float(now)

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.updated_at = now
        if elapsed:
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)

    def consume(self, now: Optional[float] = None, amount: float = 1.0) -> Tuple[bool, float]:
        """尝试取走 `amount` 个令牌。

        返回 (是否放行, 需要等待的秒数)；被拒绝时桶已被补满到当前时刻的真实值。
        """
        now = time.monotonic() if now is None else float(now)
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        missing = amount - self.tokens
        wait = missing / self.rate if self.rate > 0 else float("inf")
        return False, wait


class RateLimiter:
    """按身份（token 或客户端 IP）分桶的令牌桶限流器。"""

    def __init__(
        self,
        rps: float,
        burst: int,
        max_buckets: int = DEFAULT_MAX_BUCKETS,
    ):
        self.rps = float(rps)
        self.burst = int(burst)
        self.max_buckets = int(max_buckets)
        self._buckets: Dict[Any, TokenBucket] = {}

    @property
    def enabled(self) -> bool:
        return self.rps > 0 and self.burst > 0

    def allow(self, key: Any, now: Optional[float] = None) -> Tuple[bool, float]:
        """返回 (是否放行, Retry-After 秒数)。限流未开启时恒放行。"""
        if not self.enabled:
            return True, 0.0
        now = time.monotonic() if now is None else float(now)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self.rps, self.burst, now=now)
            self._buckets[key] = bucket
            self._evict_if_needed()
        return bucket.consume(now)

    def _evict_if_needed(self) -> None:
        """控制内存：优先淘汰「已回满」的桶——它们不含任何待恢复状态，删掉等价于重置空闲客户端。"""
        if len(self._buckets) <= self.max_buckets:
            return
        for key in [k for k, b in self._buckets.items() if b.tokens >= b.capacity][: self.max_buckets // 4]:
            self._buckets.pop(key, None)
        # 极端情况（全部都在限流中）仍超限：按最早创建者淘汰，宁可放宽也不让内存无界增长
        while len(self._buckets) > self.max_buckets:
            self._buckets.pop(next(iter(self._buckets)))


def _bearer_token(headers: List[Tuple[bytes, bytes]]) -> str:
    """从 ASGI headers 里取出 Bearer token；没有则返回空串。"""
    for name, value in headers:
        if name.lower() == b"authorization":
            raw = value.decode("latin-1").strip()
            if raw[:7].lower() == "bearer ":
                return raw[7:].strip()
            return ""
    return ""


def _json_body(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def _send_json(send: Callable, status: int, payload: Dict[str, Any], headers=None) -> None:
    """直接按 ASGI 协议回一个 JSON 响应（不依赖 Starlette，便于单测）。"""
    body = _json_body(payload)
    raw_headers = [
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(body)).encode("latin-1")),
    ]
    for name, value in headers or []:
        raw_headers.append((name.encode("latin-1"), str(value).encode("latin-1")))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


class SecurityMiddleware:
    """纯 ASGI 中间件：鉴权 + 限流 + 请求指标。只处理 `http` scope，其余（lifespan 等）原样透传。"""

    def __init__(self, app: Callable, settings: SecuritySettings, metrics=None):
        self.app = app
        self.settings = settings
        self.metrics = metrics
        self.limiter = RateLimiter(rps=settings.rps, burst=settings.burst_capacity)

    # ---------- 身份 ----------

    def _client_ip(self, scope: MutableMapping[str, Any]) -> str:
        client = scope.get("client") or ("unknown", 0)
        return str(client[0]) if isinstance(client, (tuple, list)) and client else "unknown"

    def _is_exempt(self, path: str) -> bool:
        if path in self.settings.exempt_paths:
            return True
        # 容忍 `/healthz/` 这类带尾斜杠的写法（k8s 探针配置里很常见）
        return bool(path) and path.rstrip("/") in self.settings.exempt_paths

    # ---------- 主流程 ----------

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        token = _bearer_token(list(scope.get("headers") or []))

        if not self._is_exempt(path):
            identity = ("token", token) if token else ("ip", self._client_ip(scope))
            allowed, retry_after = self.limiter.allow(identity)
            if not allowed:
                if self.metrics:
                    self.metrics.record_rate_limited()
                retry_seconds = max(1, int(math.ceil(retry_after))) if math.isfinite(retry_after) else 60
                logger.warning(f"限流拒绝 {identity[0]}={identity[1]} {scope.get('method')} {path}")
                await self._reject(
                    send,
                    429,
                    {
                        "error": "rate_limited",
                        "detail": (
                            f"超过限流阈值（{self.settings.rps:g} rps / burst "
                            f"{self.settings.burst_capacity}），请稍后重试"
                        ),
                        "retry_after": retry_seconds,
                    },
                    headers=[("Retry-After", retry_seconds)],
                )
                return

            if self.settings.auth_enabled and not hmac.compare_digest(token, self.settings.token):
                if self.metrics:
                    self.metrics.record_auth_failure()
                logger.warning(f"鉴权失败 {identity[0]}={identity[1]} {scope.get('method')} {path}")
                await self._reject(
                    send,
                    401,
                    {
                        "error": "unauthorized",
                        "detail": "缺少或无效的 Authorization: Bearer <token>",
                    },
                    headers=[("WWW-Authenticate", 'Bearer realm="rag-mcp-server"')],
                )
                return

        await self._observe(scope, receive, send)

    async def _observe(self, scope, receive, send) -> None:
        """放行并统计：记录响应状态码与耗时（异常按 500 计，然后原样抛出交给 ASGI 层）。"""
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if self.metrics:
                self.metrics.record_request(
                    status_holder["status"], (time.perf_counter() - started) * 1000
                )

    async def _reject(self, send, status: int, payload: Dict[str, Any], headers=None) -> None:
        """拒绝请求：回 JSON + 计入 requests_total（拒绝也是服务端返回的一次响应）。"""
        started = time.perf_counter()
        await _send_json(send, status, payload, headers=headers)
        if self.metrics:
            self.metrics.record_request(status, (time.perf_counter() - started) * 1000)
