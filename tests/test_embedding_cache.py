"""pytest：embedding 缓存键的行为契约。

**不需要真实 BGE 模型**：`EmbeddingEngine(config, model=假模型)` 可以注入一个假编码器，
因此这些用例在 CI runner（没有本机模型缓存）上也能跑。

为什么这组用例重要（回归防护）：
早期实现的缓存键 = 纯文本 MD5，**不含模型名**。于是改了 `embedding.model_name` 之后
旧缓存照样命中 —— 拿回来的是**旧模型**的向量，检索指标却看着「一切正常」。
这是典型的静默污染：实验数字看着对，实际跑的是另一个模型。
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import load_config  # noqa: E402
from rag.embedding import EmbeddingEngine, model_slug  # noqa: E402

MODEL_A = "BAAI/bge-small-zh-v1.5"
MODEL_B = "BAAI/bge-base-zh-v1.5"


class _FakeModel:
    """假编码器：产出**与模型名绑定**的确定性向量。

    为什么要跟模型名绑定：只有这样，「B 引擎其实读到了 A 引擎的缓存」才会被测出来 ——
    如果假模型对所有模型返回同一个向量，两边结果看起来一样，回归就测不出来。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.seed = (hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8])
        self.seed_value = int(self.seed, 16) / 0xFFFFFFFF

    def encode(self, texts, **kwargs):
        return np.array(
            [[self.seed_value, float(len(text))] for text in texts],
            dtype=np.float32,
        )


def _engine(model_name: str, cache_dir: Path) -> EmbeddingEngine:
    config = load_config()
    config["embedding"]["model_name"] = model_name
    config["embedding"]["cache_dir"] = str(cache_dir)
    return EmbeddingEngine(config, model=_FakeModel(model_name))


def test_same_text_different_model_does_not_share_cache_file(tmp_path):
    """同一段文本、两个不同 model_name → 必须是**两个不同的缓存文件**。

    旧实现（缓存键 = 纯文本 MD5、文件名 = 纯哈希）在这个场景下会命中同一个文件：
    第二个引擎直接拿第一个模型的向量当自己的结果用 —— 指标却看着「一切正常」。
    """
    chunks = [{"text": "缓存键隔离探针：同一段文本"}]

    engine_a = _engine(MODEL_A, tmp_path)
    engine_b = _engine(MODEL_B, tmp_path)

    vector_a = engine_a.encode(chunks)
    vector_b = engine_b.encode(chunks)

    files = sorted(path.name for path in tmp_path.iterdir())
    assert len(files) == 2, f"两个模型应各写一份缓存，实际只有：{files}"
    assert engine_a.cache_misses == 1 and engine_a.cache_hits == 0
    assert engine_b.cache_misses == 1 and engine_b.cache_hits == 0
    # 关键断言：两个模型拿到的**不是同一份向量**（否则就是拿了旧模型的结果冒充新模型）
    assert not np.allclose(vector_a, vector_b)


def test_cache_key_differs_by_model_name_but_is_stable(tmp_path):
    """缓存键本身：换模型必变、同一模型恒定。"""
    engine_a = _engine(MODEL_A, tmp_path)
    engine_b = _engine(MODEL_B, tmp_path)
    texts = ["键稳定性探针"]

    assert engine_a._cache_key(texts) != engine_b._cache_key(texts)
    assert engine_a._cache_key(texts) == engine_a._cache_key(texts)
    # 文本顺序不同 → 键不同（避免多段文本错配）
    assert engine_a._cache_key(["a", "b"]) != engine_a._cache_key(["b", "a"])


def test_cache_key_formula_is_pinned(tmp_path):
    """键公式钉死：md5(模型名 + "\\n" + "|||".join(文本))。

    为什么钉死：这个公式必须和 rag-eval-platform 侧**逐字节一致**（那边 2026-10-20
    做了同样的修正），两边才能算出同一个键。这里写死断言，改公式就会红。
    """
    engine = _engine(MODEL_A, tmp_path)
    texts = ["甲", "乙"]
    blob = f"{MODEL_A}\n" + "|||".join(texts)
    expected = hashlib.md5(blob.encode("utf-8")).hexdigest()
    assert engine._cache_key(texts) == expected
    # 单段文本时就是 md5(模型名 + "\n" + 文本)
    single = hashlib.md5(f"{MODEL_A}\n甲".encode("utf-8")).hexdigest()
    assert engine._cache_key(["甲"]) == single


def test_same_model_hits_cache_on_second_encode(tmp_path):
    """同一个模型、同一段文本：第二次必须命中缓存（缓存本身没被改坏）。"""
    engine = _engine(MODEL_A, tmp_path)
    chunks = [{"text": "同一模型重复编码"}]

    first = engine.encode(chunks)
    second = engine.encode(chunks)

    assert engine.cache_misses == 1
    assert engine.cache_hits == 1
    assert np.allclose(first, second)


def test_cache_file_name_is_model_scoped_and_readable(tmp_path):
    """缓存文件名带模型标识：人一眼能看出这份向量是哪份模型算的。"""
    engine = _engine(MODEL_A, tmp_path)
    engine.encode([{"text": "文件名可读性"}])

    names = [path.name for path in tmp_path.iterdir()]
    assert len(names) == 1
    assert names[0].startswith(f"{model_slug(MODEL_A)}-") and names[0].endswith(".npy")


def test_legacy_cache_without_model_name_is_ignored(tmp_path):
    """旧版缓存键（纯文本 MD5）升级后**不再命中** → 自动重算。

    这就是「不兼容旧缓存」的有意取舍：换了键，旧文件自然作废，
    代价只是第一次重算一遍向量；好处是绝不会把旧模型的向量当成新模型的结果。
    """
    engine = _engine(MODEL_A, tmp_path)
    text = "旧缓存探针"
    legacy_key = hashlib.md5(text.encode("utf-8")).hexdigest()  # 旧算法：只有文本
    np.save(tmp_path / f"{legacy_key}.npy", np.zeros((1, 2), dtype=np.float32))

    engine.encode([{"text": text}])

    assert engine.cache_misses == 1
    assert engine.cache_hits == 0
    # 旧文件没有被删（缓存是优化不是权威数据），但也不再被使用
    assert (tmp_path / f"{legacy_key}.npy").exists()


def test_shape_mismatch_cache_is_treated_as_miss(tmp_path):
    """形状不符的缓存 → 按未命中重算（保留原有的自愈逻辑）。"""
    engine = _engine(MODEL_A, tmp_path)
    chunks = [{"text": "形状不符探针"}]
    path = engine._cache_path([c["text"] for c in chunks])
    np.save(path, np.zeros((3, 2), dtype=np.float32))  # 行数对不上

    engine.encode(chunks)

    assert engine.cache_misses == 1
    assert engine.cache_hits == 0


def test_corrupted_cache_file_is_treated_as_miss(tmp_path):
    """半截/损坏的缓存文件 → 按未命中重算并覆盖（保留原有的自愈逻辑）。"""
    engine = _engine(MODEL_A, tmp_path)
    chunks = [{"text": "损坏文件探针"}]
    path = engine._cache_path([c["text"] for c in chunks])
    path.write_bytes(b"not-a-numpy-file")

    vector = engine.encode(chunks)

    assert vector.shape == (1, 2)
    assert engine.cache_misses == 1
    assert engine.cache_hits == 0
    # 覆盖成了合法文件
    assert np.load(path).shape == (1, 2)


def test_model_slug_is_filesystem_safe():
    """模型名里的 '/' 等字符不能进文件名（否则会被当成目录分隔符）。"""
    slug = model_slug("BAAI/bge-small-zh-v1.5")
    assert slug == "BAAI_bge-small-zh-v1.5"
    assert "/" not in slug and "\\" not in slug


if __name__ == "__main__":  # 允许直接 python tests\test_embedding_cache.py 跑
    raise SystemExit(pytest.main([__file__, "-v"]))
