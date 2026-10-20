"""pytest：耗时指标的 Prometheus 语义契约（离线可跑，不依赖模型）。

背景（这组用例在防什么）：
耗时指标曾经被声明成 `summary`，但分位数取自**最近 2048 次采样的滑动窗口**。
summary 的语义是「对进程启动以来的全部观测求分位数」，滑动窗口的分位数**会随时间下降** ——
拿 `rate()` 做面板会算出误导性的数字。
现在的口径：`rag_mcp_request_duration_ms` 是**直方图**（桶计数单调），
滑动窗口分位数另起一个名字 `rag_mcp_request_duration_ms_recent`（untyped，只供人看）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.metrics import DEFAULT_DURATION_BUCKETS, Metrics  # noqa: E402


def _bucket_value(text: str, le: str) -> int:
    """从渲染结果里取某个 le 桶的计数。"""
    for line in text.splitlines():
        if line.startswith(f'rag_mcp_request_duration_ms_bucket{{le="{le}"}}'):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError(f"渲染结果里没有 le={le} 的桶：\n{text}")


def test_duration_metric_is_declared_as_histogram():
    metrics = Metrics()
    metrics.record_request(200, 12.0)
    text = metrics.render_prometheus()

    assert "# TYPE rag_mcp_request_duration_ms histogram" in text
    # 滑动窗口那套不能还挂在同一个 family 上（否则类型语义又是错的）
    assert "rag_mcp_request_duration_ms{quantile=" not in text


def test_help_text_gives_the_aggregated_quantile_query():
    """HELP 里推荐的 PromQL 必须是**聚合后**的分位数写法。

    `histogram_quantile(0.95, rate(..._bucket[5m]))` 少了 `sum by (le)`：
    单实例碰巧能算，多实例时每个 `le` 有多条序列 → Prometheus 直接报错。
    这条断言把「正确写法」钉死在代码里，防止以后又被改回缺聚合的版本。
    """
    metrics = Metrics()
    metrics.record_request(200, 12.0)
    text = metrics.render_prometheus()

    help_line = next(
        line for line in text.splitlines() if line.startswith("# HELP rag_mcp_request_duration_ms ")
    )
    assert "histogram_quantile(0.95, sum(rate(rag_mcp_request_duration_ms_bucket[5m])) by (le))" in help_line
    assert "histogram_quantile(0.95, rate(" not in help_line


def test_histogram_buckets_are_cumulative_and_match_count():
    metrics = Metrics()
    for duration in (0.5, 3.0, 7.0, 30.0, 200.0, 30000.0):
        metrics.record_request(200, duration)

    text = metrics.render_prometheus()

    assert _bucket_value(text, "1") == 1       # 只有 0.5ms
    assert _bucket_value(text, "5") == 2       # 0.5 + 3.0
    assert _bucket_value(text, "10") == 3      # 再加 7.0
    assert _bucket_value(text, "50") == 4      # 再加 30.0
    assert _bucket_value(text, "1000") == 5    # 再加 200.0
    assert _bucket_value(text, "+Inf") == 6    # 再加 30000.0（超出所有有限桶）
    # +Inf 必须等于总观测数（= _count）
    assert _bucket_value(text, "+Inf") == 6
    assert "rag_mcp_request_duration_ms_count 6" in text
    assert "rag_mcp_request_duration_ms_sum 30240.5" in text


def test_histogram_bucket_counts_never_decrease():
    """直方图桶是**单调**的：窗口滑走也不会让计数变小（这是相对 summary 的关键改进）。"""
    metrics = Metrics(max_samples=100)
    for i in range(1, 101):
        metrics.record_request(200, float(i))  # 1..100 ms
    before = metrics.snapshot()["request_duration_ms"]["buckets"]

    for _ in range(1000):  # 全是 1ms 的快请求，把慢请求挤出「最近 N 次」窗口
        metrics.record_request(200, 1.0)
    after = metrics.snapshot()["request_duration_ms"]["buckets"]

    for le, value in after.items():
        assert value >= before[le], f"桶 {le} 的计数下降了：{before[le]} → {value}"
    assert after["+Inf"] == 1100
    assert metrics.snapshot()["request_duration_ms"]["count"] == 1100


def test_recent_quantiles_are_declared_untyped_and_documented():
    metrics = Metrics()
    metrics.record_request(200, 5.0)
    text = metrics.render_prometheus()

    assert "# TYPE rag_mcp_request_duration_ms_recent untyped" in text
    assert 'rag_mcp_request_duration_ms_recent{quantile="0.5"}' in text
    assert 'rag_mcp_request_duration_ms_recent{quantile="0.95"}' in text
    assert 'rag_mcp_request_duration_ms_recent{quantile="0.99"}' in text
    # HELP 里必须写明「滑动窗口」，免得有人拿它做面板
    help_line = next(
        line for line in text.splitlines() if line.startswith("# HELP rag_mcp_request_duration_ms_recent")
    )
    assert "滑动窗口" in help_line


def test_recent_quantiles_follow_the_sliding_window():
    """滑动窗口语义：慢请求滑出窗口后，最近分位数会**变小**（这正是它不能当 summary 的原因）。"""
    metrics = Metrics(max_samples=100)
    for i in range(1, 101):
        metrics.record_request(200, float(i))
    assert metrics.snapshot()["request_duration_ms"]["p50"] == 50.0

    for _ in range(100):
        metrics.record_request(200, 1.0)  # 100 个 1ms 把窗口整体换掉
    snapshot = metrics.snapshot()["request_duration_ms"]
    assert snapshot["p50"] == 1.0  # 窗口滑走了
    assert snapshot["count"] == 200  # 但总计数不减
    assert len(metrics._durations) == 100  # 采样窗口有上限，内存不随请求数增长


def test_default_buckets_cover_fast_and_slow_paths():
    """桶边界要覆盖「命中缓存的快请求」与「慢请求」两端。"""
    assert DEFAULT_DURATION_BUCKETS[0] <= 1.0
    assert DEFAULT_DURATION_BUCKETS[-1] >= 10_000
    assert list(DEFAULT_DURATION_BUCKETS) == sorted(DEFAULT_DURATION_BUCKETS)


if __name__ == "__main__":  # 允许直接 python tests\test_metrics.py 跑
    raise SystemExit(pytest.main([__file__, "-v"]))
