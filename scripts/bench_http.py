"""压测脚本：对已运行的 HTTP MCP Server 做并发检索压测，输出 p50/p95/p99 / QPS / 错误率。

与 `mcp_http_client_demo.py`（1 个客户端、验功能）的区别：本脚本要的是**量**——
N 个并发客户端复用 **长连接**（官方 `streamable_http_client`）持续打 `search_knowledge`，
统计延迟分位数与吞吐，用来回答「这个服务在生产里能扛多少 QPS、尾延迟多高」。

为什么坚持用官方 MCP 客户端而不是裸 HTTP 压测：
- 裸打 HTTP 只能测「协议解析」，测不到 MCP 握手/会话管理/工具调用的真实开销；
- 长连接（keep-alive + 会话复用）是生产客户端的真实形态，每请求新建连接的测法会把
  三次握手/TLS 开销算进结果里，得到的 QPS 没有参考价值。

先启动服务（另开终端，生产形态建议同时开鉴权）：
    .venv\\Scripts\\python.exe server.py --transport streamable-http --port 8000
    # 或带鉴权：set RAG_MCP_AUTH_TOKEN=xxx && .venv\\Scripts\\python.exe server.py --transport streamable-http

再压测：
    .venv\\Scripts\\python.exe scripts\\bench_http.py --concurrency 20 --requests 100
    .venv\\Scripts\\python.exe scripts\\bench_http.py -n 50 -c 20 --token $env:RAG_MCP_AUTH_TOKEN

输出：并发数 / 总请求数 / 成功数 / 错误率 / QPS / p50 / p95 / p99 / max，以及服务端 /metrics 的
`tool_calls_total` 与 requests_total（客户端成功数 vs 服务端计数交叉验证——两个数字对得上，
才说明「压出来的 QPS」不是客户端在自欺欺人）。

注意：服务端若开了限流（server.rate_limit.rps），超过阈值的请求会被 429 拒掉，
此时错误率会上升、QPS 会被限流阈值封顶——这本身就是被测对象的一部分，务必在报告里写清楚。
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request
from typing import Dict, List
from urllib.parse import urlsplit

DEFAULT_QUERY = "LoRA 微调的原理是什么？"


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCP HTTP 传输压测（并发 x 次数，统计 p50/p95/p99 与 QPS）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=None, help="完整端点 URL；缺省按 host/port/path 拼")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp", help="streamable-http 端点路径")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="并发客户端（长连接）数")
    parser.add_argument("-n", "--requests", type=int, default=100, help="总请求数（在所有并发客户端间分摊）")
    parser.add_argument("--tool", default="search_knowledge", help="压测的工具名")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="检索用的 query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--token", default=None, help="Bearer token（缺省读环境变量 RAG_MCP_AUTH_TOKEN）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单请求超时（秒）")
    parser.add_argument("--warmup", type=int, default=3, help="正式计时前的预热请求数（排除首调用噪声）")
    parser.add_argument("--no-metrics", action="store_true", help="不抓取服务端 /metrics 做交叉验证")
    parser.add_argument("--json", action="store_true", help="额外输出一行 JSON（便于脚本化对比）")
    return parser.parse_args(argv)


def _build_url(args: argparse.Namespace) -> str:
    if args.url:
        return args.url
    path = args.path if args.path.startswith("/") else "/" + args.path
    return f"http://{args.host}:{args.port}{path}"


def _fetch_metrics(base_url: str, timeout: float = 5.0) -> Dict[str, float]:
    """抓服务端 /metrics（免鉴权）并解析成 {样本名: 值}；解析失败返回空表（不阻断压测结论）。"""
    try:
        # 不走系统代理：本机全局代理会把 127.0.0.1 的请求也代理走 → 502
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(base_url.rstrip("/") + "/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:  # 指标抓不到不影响压测本身
        print(f"[warn] /metrics 抓取失败（{type(exc).__name__}: {exc}），跳过交叉验证")
        return {}

    parsed: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            parsed[name.strip()] = float(value)
        except ValueError:
            continue
    return parsed


def _percentile(sorted_values: List[float], q: float) -> float:
    """最近秩法（nearest-rank）：与 rag/metrics.py 的口径一致，便于两边数字对齐。"""
    if not sorted_values:
        return 0.0
    rank = max(1, min(len(sorted_values), int(round(q * len(sorted_values)))))
    return sorted_values[rank - 1]


async def _worker(
    url: str,
    args: argparse.Namespace,
    queue: "asyncio.Queue[int]",
    results: List[dict],
    authorization: str | None,
    barrier: "asyncio.Barrier",
) -> None:
    """一个并发客户端：建立一条长连接 + 一个 MCP 会话，然后循环领取任务直到队列空。"""
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"Authorization": authorization} if authorization else {}
    limits = httpx2.Limits(
        max_connections=max(4, args.concurrency),
        max_keepalive_connections=max(4, args.concurrency),
    )
    # trust_env=False：忽略系统代理（含 http.proxy=127.0.0.1:7897），否则本机请求会被代理走
    async with httpx2.AsyncClient(
        headers=headers, limits=limits, timeout=args.timeout, trust_env=False
    ) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 预热：把首个请求的模型/JIT/连接开销排除在统计之外
                for _ in range(args.warmup):
                    try:
                        await session.call_tool(args.tool, {"query": args.query, "top_k": args.top_k})
                    except Exception:
                        break

                # 所有客户端预热完毕再一起开跑：否则「先连上的客户端」会边跑边等别人连，
                # 墙钟时间里混进了建连耗时，算出来的 QPS 会明显偏低
                await barrier.wait()

                while True:
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    started = time.perf_counter()
                    error = None
                    matched = None
                    try:
                        result = await session.call_tool(
                            args.tool, {"query": args.query, "top_k": args.top_k}
                        )
                        payload = json.loads(result.content[0].text)
                        if "error" in payload:
                            error = f"tool_error: {payload['error'][:80]}"
                        else:
                            matched = payload.get("matched")
                    except Exception as exc:  # 网络/协议/限流异常都算失败，但要分类统计
                        error = f"{type(exc).__name__}: {str(exc)[:120]}"
                    finally:
                        queue.task_done()

                    results.append(
                        {
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                            "error": error,
                            "matched": matched,
                        }
                    )


async def _run(args: argparse.Namespace) -> int:
    url = _build_url(args)
    base_url = "{0.scheme}://{0.netloc}".format(urlsplit(url))  # /healthz、/metrics 挂在根路径下
    token = args.token or os.environ.get("RAG_MCP_AUTH_TOKEN")
    authorization = f"Bearer {token}" if token else None

    if args.requests < args.concurrency:
        print(f"[warn] 总请求数({args.requests}) < 并发数({args.concurrency})，实际并发受请求数限制")

    queue: "asyncio.Queue[int]" = asyncio.Queue()
    for index in range(args.requests):
        queue.put_nowait(index)

    results: List[dict] = []
    printed_auth = "（带 Bearer token）" if authorization else "（无鉴权）"
    print(
        f"[bench] {url} {printed_auth}\n"
        f"[bench] 工具={args.tool} query={args.query!r} 并发={args.concurrency} 请求={args.requests} "
        f"预热/客户端={args.warmup}"
    )

    # 压测前先抓一次 /metrics：服务端计数器是进程级累计值，只有取**差值**才能和本次压测对上
    metrics_before = {} if args.no_metrics else _fetch_metrics(base_url)

    # 并发度 + 1：多方里额外那一方是主协程，用它来卡「所有客户端预热完毕」的时刻
    barrier = asyncio.Barrier(args.concurrency + 1)
    tasks = [
        asyncio.create_task(_worker(url, args, queue, results, authorization, barrier))
        for _ in range(args.concurrency)
    ]
    try:
        # 加超时保护：若某个客户端预热阶段就失败退出，主协程不能在这里永久等待
        await asyncio.wait_for(barrier.wait(), timeout=args.timeout)
    except (TimeoutError, asyncio.TimeoutError):
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        print(
            f"[FAIL] {args.timeout:.0f}s 内仍有客户端没完成连接/预热；"
            "检查服务是否已启动、--url/--port 是否一致、token 是否正确",
            file=sys.stderr,
        )
        return 1
    wall_started = time.perf_counter()  # 计时从「全员预热完毕」开始，QPS 不含建连/预热
    await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - wall_started

    if not results:
        print("[FAIL] 没有任何请求完成：检查服务是否已启动、token 是否正确", file=sys.stderr)
        return 1

    latencies = sorted(r["elapsed_ms"] for r in results)
    ok = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    error_rate = len(errors) / len(results) * 100
    qps = len(results) / wall_seconds if wall_seconds > 0 else 0.0

    print(
        "\n=== 客户端侧结果 ===\n"
        f"总请求 {len(results)} | 成功 {len(ok)} | 失败 {len(errors)} | 错误率 {error_rate:.2f}%\n"
        f"墙钟 {wall_seconds:.2f}s | QPS {qps:.1f}（成功 {len(ok) / wall_seconds:.1f}）\n"
        f"延迟(ms)  p50 {_percentile(latencies, 0.50):.2f}  p95 {_percentile(latencies, 0.95):.2f}  "
        f"p99 {_percentile(latencies, 0.99):.2f}  max {latencies[-1]:.2f}  "
        f"mean {statistics.fmean(latencies):.2f}"
    )

    # 错误分类（429 限流 / 超时 / 协议错误要分开看，否则没法定位瓶颈）
    if errors:
        buckets: Dict[str, int] = {}
        for item in errors:
            key = item["error"].split(":")[0]
            if "429" in item["error"]:
                key = "429 rate_limited"
            buckets[key] = buckets.get(key, 0) + 1
        print("错误分类：" + "；".join(f"{k} x{v}" for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])))
        print("       ↑ 若是 429：说明撞到了服务端限流阈值（server.rate_limit.rps），不是性能问题")
        # 打印前 3 条原始错误：压测报告里最需要的就是「到底报了什么」，而不是只有类别
        for item in errors[:3]:
            print(f"       样例错误：{item['error']}")

    matched_values = [r["matched"] for r in ok if r["matched"] is not None]
    if matched_values:
        print(f"检索命中：平均 {statistics.fmean(matched_values):.1f} 条 / 成功请求")

    if not args.no_metrics:
        metrics = _fetch_metrics(base_url)
        if metrics:
            def delta(name: str) -> float:
                return metrics.get(name, 0) - metrics_before.get(name, 0)

            tool_sample = f'rag_mcp_tool_calls_total{{tool="{args.tool}"}}'
            tool_calls = delta(tool_sample)
            print(
                "\n=== 服务端侧（/metrics，本次压测的增量）===\n"
                f'tool_calls_total{{tool="{args.tool}"}} 增量 = {tool_calls:.0f}'
                f" | errors_total 增量 = {delta('rag_mcp_errors_total'):.0f}"
                f" | rate_limited_total 增量 = {delta('rag_mcp_rate_limited_total'):.0f}"
                f" | auth_failures_total 增量 = {delta('rag_mcp_auth_failures_total'):.0f}"
            )
            expected = len(results) + args.warmup * args.concurrency
            tolerance = max(2, expected * 0.02)
            if abs(tool_calls - expected) > tolerance:
                print(
                    f"       ⚠ 客户端统计 {expected} 次 vs 服务端增量 {tool_calls:.0f} 次不一致："
                    "可能有请求在服务端被限流/中断，或预热请求数与实际不符"
                )
            else:
                print(
                    f"       ✓ 客户端 {expected} 次 ≈ 服务端 {tool_calls:.0f} 次："
                    "两边计数一致，压测结果可信"
                )

    if args.json:
        print(
            json.dumps(
                {
                    "url": url,
                    "tool": args.tool,
                    "concurrency": args.concurrency,
                    "requests": len(results),
                    "success": len(ok),
                    "error_rate": round(error_rate, 3),
                    "qps": round(qps, 2),
                    "wall_seconds": round(wall_seconds, 3),
                    "latency_ms": {
                        "p50": round(_percentile(latencies, 0.50), 3),
                        "p95": round(_percentile(latencies, 0.95), 3),
                        "p99": round(_percentile(latencies, 0.99), 3),
                        "max": round(latencies[-1], 3),
                        "mean": round(statistics.fmean(latencies), 3),
                    },
                },
                ensure_ascii=False,
            )
        )
    return 0 if ok else 1


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:
        print(f"\n[FAIL] 压测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "       请确认服务已启动，且传输方式/端口一致，例如：\n"
            "       .venv\\Scripts\\python.exe server.py --transport streamable-http --port 8000",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
