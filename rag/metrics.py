"""进程内指标收集 + Prometheus 文本导出（零额外依赖）。

设计取舍（为什么不用 prometheus_client）：
- 需要暴露的指标只有 4 类，用几十行就能算清楚；prometheus_client 会多带来一个运行时依赖，
  而且它的 `Summary` 只支持客户端侧分位数估算（`_sum/_count/quantile`），本项目也不需要更复杂的能力；
- 零依赖意味着**任何环境装完 requirements.txt 就能 /metrics**，不会因为指标库版本引发启动失败
  （这是「可观测性不能成为故障源」的取舍：指标挂了不该拖垮服务）；
- 代价：没有多进程聚合。多副本部署时应由网关/边车统一采集（见 README §12）。

指标清单（名字与 README 的表格一一对应）：
- `rag_mcp_requests_total{status}`   —— HTTP 请求总数（按响应状态码分组）
- `rag_mcp_tool_calls_total{tool}`   —— 工具调用总数（按工具名分组）
- `rag_mcp_errors_total`             —— 错误总数（HTTP 5xx + 工具异常）
- `rag_mcp_request_duration_ms`      —— 请求耗时**直方图**（`_bucket{le=…}` / `_sum` / `_count`）
- `rag_mcp_request_duration_ms_recent{quantile}` —— 最近 N 次采样的耗时分位数（**滑动窗口**，仅供人看）
- `rag_mcp_auth_failures_total` / `rag_mcp_rate_limited_total` —— 鉴权失败 / 被限流拒绝次数
- `rag_mcp_prompt_injection_findings_total{rule}` —— 检索结果里命中的提示注入可疑模式次数
- `rag_mcp_uptime_seconds`           —— 进程内指标表存活时长（用于判断是否重启过）

线程安全：MCP 的工具处理可能在别的线程执行，因此所有写入都过一把可重入之外的普通互斥锁
（加锁粒度极小、只在计数与 append 时持有，性能影响可忽略）。

## 为什么耗时指标是 Histogram 而不是 Summary

早期实现把耗时声明成 Prometheus `summary`，但分位数取自**最近 2048 次采样的滑动窗口**。
这两件事是冲突的：summary 的语义是「对进程启动以来的**全部**观测求分位数」，
`_sum`/`_count` 是单调计数器；而滑动窗口里的分位数**会随时间下降**
（慢请求滑出窗口后 p99 会变小），拿 `rate()` 去做面板会算出误导性的数字。

现在的口径（两者并存，职责分开）：
- `rag_mcp_request_duration_ms` = **histogram**：桶计数**只增不减**（单调），
  `_sum`/`_count` 的名字与语义都不变 → 既有 `rate(rag_mcp_request_duration_ms_count[5m])`
  这类消费方**不受影响**；分位数用标准写法
  `histogram_quantile(0.95, rate(rag_mcp_request_duration_ms_bucket[5m]))` 算，且可跨实例聚合。
- `rag_mcp_request_duration_ms_recent` = **untyped**：保留原来那套滑动窗口分位数（p50/p95/p99），
  改名 + HELP 里写明「滑动窗口、值会下降、只供人看」，避免有人拿它做面板。
"""

import threading
import time
from collections import Counter, deque
from typing import Dict, List, Tuple

# Summary 保留的最近采样数：够算分位数，又不会无限增长内存
DEFAULT_MAX_SAMPLES = 2048

#: 请求耗时直方图的桶边界（毫秒）：从「本机命中缓存 ~1ms」铺到「慢请求 ~10s」。
#: 桶是**累计**的（le="5" 的计数包含所有 <=5ms 的观测），与 Prometheus histogram 语义一致。
DEFAULT_DURATION_BUCKETS: Tuple[float, ...] = (
    1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000,
)

# Prometheus 文本格式的 HELP 文案（中文，面试官/协作者都能读懂指标含义）
_HELP = {
    "requests_total": "HTTP 请求总数（按响应状态码分组）",
    "tool_calls_total": "MCP 工具调用总数（按工具名分组）",
    "errors_total": "错误总数（HTTP 5xx 响应 + 工具内部异常）",
    "request_duration_ms": (
        "HTTP 请求耗时直方图（毫秒；_bucket 为累计桶计数、_sum/_count 为单调计数器；"
        "分位数请用 histogram_quantile(0.95, rate(rag_mcp_request_duration_ms_bucket[5m]))）"
    ),
    "request_duration_ms_recent": (
        "HTTP 请求耗时的最近 N 次采样分位数（毫秒；**滑动窗口**，值会随时间下降，"
        "只供人看；做面板请用上面的直方图）"
    ),
    "auth_failures_total": "鉴权失败（401）次数",
    "rate_limited_total": "被令牌桶限流拒绝（429）的次数",
    "prompt_injection_findings_total": "检索结果中命中的提示注入可疑模式次数（按规则分组）",
    "uptime_seconds": "指标表存活时长（秒）",
}


def _escape_label(value: str) -> str:
    """按 Prometheus 文本格式转义 label 值（反斜杠、双引号、换行）。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _quantile(sorted_values: List[float], q: float) -> float:
    """最近秩法（nearest-rank）取分位数：样本少时也比插值更直观、可复现。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(1, min(len(sorted_values), int(round(q * len(sorted_values)))))
    return float(sorted_values[rank - 1])


def _format_bound(upper: float) -> str:
    """桶边界 → Prometheus `le` 标签文本（整数边界不带小数点：'5' 而不是 '5.0'）。"""
    if float(upper).is_integer():
        return str(int(upper))
    return str(upper)


class Metrics:
    """线程安全的进程内指标表。全局只建一个实例（`METRICS`），中间件与工具函数共用。"""

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES):
        self._lock = threading.Lock()
        self._requests: Counter = Counter()      # status -> count
        self._tool_calls: Counter = Counter()    # tool name -> count
        self._errors = 0
        self._auth_failures = 0
        self._rate_limited = 0
        self._injection_findings: Counter = Counter()   # 规则 id -> 命中次数
        self._durations: deque = deque(maxlen=max_samples)  # 最近 N 次采样（只给「滑动窗口」分位数用）
        self._duration_sum_ms = 0.0
        self._duration_count = 0
        # 直方图桶计数（**只增不减**，这是 histogram 相对滑动窗口的关键差别）：
        # 下标与 DEFAULT_DURATION_BUCKETS 对应，末位表示「超过最大有限桶」（即 +Inf）。
        self._duration_buckets: List[int] = [0] * (len(DEFAULT_DURATION_BUCKETS) + 1)
        self._started_at = time.time()

    # ========== 写入 ==========

    @staticmethod
    def _bucket_index(duration_ms: float) -> int:
        """落在哪个桶：第一个满足 duration <= 上界的桶；都比不过则进末位（+Inf）。"""
        for idx, upper in enumerate(DEFAULT_DURATION_BUCKETS):
            if duration_ms <= upper:
                return idx
        return len(DEFAULT_DURATION_BUCKETS)

    def record_request(self, status: int, duration_ms: float) -> None:
        """记录一次 HTTP 请求：总数 + 耗时；5xx 同时计入 errors_total。"""
        with self._lock:
            self._requests[int(status)] += 1
            duration = float(duration_ms)
            self._durations.append(duration)
            self._duration_sum_ms += duration
            self._duration_count += 1
            self._duration_buckets[self._bucket_index(duration)] += 1
            if int(status) >= 500:
                self._errors += 1

    def record_tool_call(self, tool: str) -> None:
        with self._lock:
            self._tool_calls[str(tool)] += 1

    def record_error(self) -> None:
        """工具内部异常（与 HTTP 5xx 共用 errors_total 计数）。"""
        with self._lock:
            self._errors += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self._auth_failures += 1

    def record_rate_limited(self) -> None:
        with self._lock:
            self._rate_limited += 1

    def record_injection_findings(self, counts: Dict[str, int]) -> None:
        """记录检索结果里命中的提示注入可疑模式（规则 id -> 次数）。

        为什么值得单独成为指标：注入命中率是**数据面**的安全信号，
        能提前暴露「知识库里混进了被投毒的文档」，而不是等上层 Agent 被带偏才发现。
        """
        with self._lock:
            for rule, count in (counts or {}).items():
                self._injection_findings[str(rule)] += int(count)

    # ========== 读取 ==========

    def _cumulative_buckets(self) -> Dict[str, int]:
        """把「精确落桶计数」换算成 Prometheus histogram 的**累计**桶计数。

        Prometheus 约定：le="5" 的计数 = 所有 <=5ms 的观测数（而不是「恰好落在 1~5ms 的」）。
        这里从高位往低位加，最后的 "+Inf" 等于总观测数（= _count）。
        """
        cumulative: Dict[str, int] = {}
        running = 0
        for idx, upper in enumerate(DEFAULT_DURATION_BUCKETS):
            running += self._duration_buckets[idx]
            cumulative[_format_bound(upper)] = running
        cumulative["+Inf"] = running + self._duration_buckets[-1]
        return cumulative

    def uptime_seconds(self) -> float:
        return round(time.time() - self._started_at, 3)

    def snapshot(self) -> Dict:
        """结构化快照：便于单测断言与「JSON 版 /metrics」复用。"""
        with self._lock:
            requests = dict(self._requests)
            tools = dict(self._tool_calls)
            samples = sorted(self._durations)
            total = self._duration_count
            sum_ms = self._duration_sum_ms
            errors, auth_failures, rate_limited = (
                self._errors,
                self._auth_failures,
                self._rate_limited,
            )
            injection_findings = dict(self._injection_findings)
            buckets = self._cumulative_buckets()
        return {
            "requests_total": requests,
            "requests_total_sum": sum(requests.values()),
            "tool_calls_total": tools,
            "errors_total": errors,
            "auth_failures_total": auth_failures,
            "rate_limited_total": rate_limited,
            "prompt_injection_findings_total": injection_findings,
            "request_duration_ms": {
                "count": total,
                "sum": round(sum_ms, 3),
                "p50": round(_quantile(samples, 0.50), 3),
                "p95": round(_quantile(samples, 0.95), 3),
                "p99": round(_quantile(samples, 0.99), 3),
                "max_samples": self._durations.maxlen,
                # 直方图桶（累计语义）：{"1": 0, "2": 1, ..., "+Inf": 3}
                "buckets": buckets,
            },
            "uptime_seconds": self.uptime_seconds(),
        }

    def render_prometheus(self) -> str:
        """导出 Prometheus 文本格式（text/plain; version=0.0.4）。"""
        snap = self.snapshot()
        lines: List[str] = []

        def header(name: str, mtype: str) -> None:
            lines.append(f"# HELP rag_mcp_{name} {_HELP[name]}")
            lines.append(f"# TYPE rag_mcp_{name} {mtype}")

        header("requests_total", "counter")
        if snap["requests_total"]:
            for status, count in sorted(snap["requests_total"].items()):
                lines.append(f'rag_mcp_requests_total{{status="{int(status)}"}} {count}')
        else:
            # 无样本时也输出 0 值，保证仪表盘不会因为「指标不存在」而报错
            lines.append('rag_mcp_requests_total{status="none"} 0')

        header("tool_calls_total", "counter")
        if snap["tool_calls_total"]:
            for tool, count in sorted(snap["tool_calls_total"].items()):
                lines.append(
                    f'rag_mcp_tool_calls_total{{tool="{_escape_label(tool)}"}} {count}'
                )
        else:
            lines.append('rag_mcp_tool_calls_total{tool="none"} 0')

        header("errors_total", "counter")
        lines.append(f"rag_mcp_errors_total {snap['errors_total']}")

        header("request_duration_ms", "histogram")
        duration = snap["request_duration_ms"]
        for le, value in duration["buckets"].items():
            lines.append(f'rag_mcp_request_duration_ms_bucket{{le="{le}"}} {value}')
        lines.append(f"rag_mcp_request_duration_ms_sum {duration['sum']}")
        lines.append(f"rag_mcp_request_duration_ms_count {duration['count']}")

        # 滑动窗口分位数：保留给人看（原来的 summary 口径），名字里带 _recent 表明它不是累计量。
        # 声明成 untyped 而不是 summary —— summary 的语义是「进程启动以来的全部分位数」，
        # 而这里的分位数只覆盖最近 N 次采样、**会随时间下降**，用它做面板会算出误导性的数字。
        header("request_duration_ms_recent", "untyped")
        for q, key in (("0.5", "p50"), ("0.95", "p95"), ("0.99", "p99")):
            lines.append(f'rag_mcp_request_duration_ms_recent{{quantile="{q}"}} {duration[key]}')

        header("auth_failures_total", "counter")
        lines.append(f"rag_mcp_auth_failures_total {snap['auth_failures_total']}")

        header("rate_limited_total", "counter")
        lines.append(f"rag_mcp_rate_limited_total {snap['rate_limited_total']}")

        header("prompt_injection_findings_total", "counter")
        if snap["prompt_injection_findings_total"]:
            for rule, count in sorted(snap["prompt_injection_findings_total"].items()):
                lines.append(
                    f'rag_mcp_prompt_injection_findings_total{{rule="{_escape_label(rule)}"}} {count}'
                )
        else:
            # 无样本时也输出 0 值（规则 id 未知 → 用 none 占位），仪表盘不会因样本缺失报错
            lines.append('rag_mcp_prompt_injection_findings_total{rule="none"} 0')

        header("uptime_seconds", "gauge")
        lines.append(f"rag_mcp_uptime_seconds {snap['uptime_seconds']}")

        return "\n".join(lines) + "\n"


#: 进程级单例：server.py 的中间件与工具函数共用一个指标表
METRICS = Metrics()

