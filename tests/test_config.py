"""pytest：配置加载的 fail-fast 行为契约（离线可跑，不依赖模型）。

重点：`chunking.strategy` 必须**要么真生效、要么启动就报错**，
不能出现「配了 semantic 却静默按 fixed_size 跑」的静默失效
（口径与 rag/http_security.py 的 SecuritySettings 一致）。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import FixedSizeChunker  # noqa: E402
from rag.config import (  # noqa: E402
    DEFAULTS,
    SUPPORTED_CHUNKING_STRATEGIES,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_config(tmp_path: Path, strategy: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "chunking:\n"
        f'  strategy: "{strategy}"\n'
        "  chunk_size: 512\n"
        "  chunk_overlap: 50\n",
        encoding="utf-8",
    )
    return path


def test_repo_default_config_is_valid():
    """仓库自带的 config.yaml 必须能通过校验（别把自己的默认配置写坏了）。"""
    config = load_config(REPO_ROOT / "config.yaml")
    assert config["chunking"]["strategy"] == "fixed_size"


def test_supported_strategy_loads_normally(tmp_path):
    config = load_config(_write_config(tmp_path, "fixed_size"))
    assert config["chunking"]["strategy"] == "fixed_size"


def test_unsupported_strategy_fails_fast(tmp_path):
    """写了没实现的策略 → 加载时直接 ValueError，而不是静默忽略。"""
    with pytest.raises(ValueError) as excinfo:
        load_config(_write_config(tmp_path, "semantic"))
    message = str(excinfo.value)
    assert "semantic" in message
    assert "fixed_size" in message


def test_strategy_is_case_and_space_insensitive(tmp_path):
    """大小写/空格差异不算「另一个策略」—— 归一化后再校验。"""
    config = load_config(_write_config(tmp_path, " FIXED_SIZE "))
    assert config["chunking"]["strategy"].strip().lower() == "fixed_size"


def test_default_strategy_is_in_whitelist():
    assert DEFAULTS["chunking"]["strategy"] in SUPPORTED_CHUNKING_STRATEGIES
    assert SUPPORTED_CHUNKING_STRATEGIES == ("fixed_size",)


def test_chunker_rejects_unsupported_strategy_even_for_handmade_config():
    """绕过 load_config 手工拼的配置，切分器自己也要拦一道（双保险）。"""
    config = {
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 50,
        }
    }
    with pytest.raises(ValueError):
        FixedSizeChunker(config)


if __name__ == "__main__":  # 允许直接 python tests\test_config.py 跑
    raise SystemExit(pytest.main([__file__, "-v"]))
