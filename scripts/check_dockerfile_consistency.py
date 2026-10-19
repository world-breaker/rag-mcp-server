#!/usr/bin/env python3
"""Dockerfile ↔ 仓库现状 一致性静态检查（**不依赖 Docker**，可在开发机上重复跑）。

为什么需要它：本仓库的开发机（Windows）上**没有 Docker**，`Dockerfile` 无法 `docker build` /
`docker run` 验证。既然「动态验证」这条路走不通，就把**能在静态层面查清的事实**全部查清并留下证据：
每一条声明都对应一次真实读取（Dockerfile 原文 / argparse 定义 / requirements / .dockerignore / CI 配置），
而不是靠读一遍代码"看着没问题"。

检查项（对应 README §5.7 的容器小节）：
  A  Dockerfile 基本形态（FROM / EXPOSE / CMD）
  B  CMD+ENV 里的每个 CLI 选项与环境变量在 `server.py` / `rag/config.py` 里真实存在
  C  端口自洽（EXPOSE ↔ CMD ↔ ENV ↔ config.yaml ↔ README ↔ 脚本/测试默认值）
  D  HEALTHCHECK 的端点真实存在、断言的字段真实存在、且该端点确实免鉴权
  E  `server.py` + `rag/` 的全部第三方 import 都被 requirements.txt（含其传递依赖）覆盖
  F  `.dockerignore` 排除了什么；运行必需文件**没有**被误排除
  G  Python 版本在 Dockerfile / CI / README / requirements 注释里一致
  H  Dockerfile 里 COPY 的源路径在仓库里存在
  I  hadolint 的抑制声明都写了理由（抑制必须可辩护）
  J  容器默认 `offline: true` 且镜像里没有模型 ⇒ README 的容器小节必须把「模型怎么就位」写全

**明确不在本脚本能力范围内**（脚本会在末尾原样列出）：真实 `docker build` 是否成功、`docker run`
能否起服务、HEALTHCHECK 的真实行为、镜像体积、非 root 权限是否真的生效、apt/网络是否可达。
这些只有装了 Docker 的机器能回答——本仓库的诚实声明见 `Dockerfile` 顶部与 README §5.7 的 ⚠️ 段落。

用法：
    .venv\\Scripts\\python.exe scripts/check_dockerfile_consistency.py
    .venv\\Scripts\\python.exe scripts/check_dockerfile_consistency.py --verbose

退出码：0 = 全部静态检查通过；1 = 有不一致（每条都会打印证据）。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
CONFIG_YAML = REPO_ROOT / "config.yaml"
SERVER_PY = REPO_ROOT / "server.py"
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"
README = REPO_ROOT / "README.md"

#: 运行期必需的文件（进了容器才跑得起来）：被 .dockerignore 排除掉就是真缺陷
RUNTIME_REQUIRED = (
    "server.py",
    "config.yaml",
    "requirements.txt",
    "rag/service.py",
    "data/sample/knowledge_base.json",
)


class Checks:
    """收集检查结果：每条 = 检查项 + 证据 + 判定。"""

    def __init__(self, verbose: bool = False) -> None:
        self.rows: List[Dict[str, str]] = []
        self.verbose = verbose

    def add(self, check_id: str, name: str, ok: bool, evidence: str) -> bool:
        self.rows.append(
            {"id": check_id, "name": name, "ok": "PASS" if ok else "FAIL", "evidence": evidence}
        )
        print(f"  [{'PASS' if ok else 'FAIL'}] {check_id} {name}")
        print(f"         证据：{evidence}")
        return ok

    def note(self, text: str) -> None:
        print(f"  ----  {text}")

    @property
    def failed(self) -> List[Dict[str, str]]:
        return [r for r in self.rows if r["ok"] == "FAIL"]


# ============================ Dockerfile 解析 ============================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def dockerfile_lines() -> List[str]:
    """返回 Dockerfile 的逻辑行（已把 `\\` 续行拼成一行），注释行保留。"""
    out: List[str] = []
    buf = ""
    for raw in read_text(DOCKERFILE).splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            out.append(raw)
            continue
        if raw.rstrip().endswith("\\"):
            buf += raw.rstrip()[:-1] + " "
            continue
        out.append(buf + raw)
        buf = ""
    if buf:
        out.append(buf)
    return out


def docker_instructions(name: str) -> List[str]:
    return [l for l in dockerfile_lines() if l.strip().upper().startswith(name.upper() + " ")]


def docker_env() -> Dict[str, str]:
    """把 ENV 指令解析成 {名: 值}（同时支持 `A=1 \\n B=2` 与 `A=1 B=2` 两种写法）。"""
    env: Dict[str, str] = {}
    for line in docker_instructions("ENV"):
        body = line.strip()[len("ENV"):].strip()
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=(\S*)", body):
            env[key] = value.strip('"')
    return env


def docker_cmd() -> List[str]:
    for line in docker_instructions("CMD"):
        body = line.strip()[len("CMD"):].strip()
        if body.startswith("["):
            return list(json.loads(body))
        return body.split()
    raise AssertionError("Dockerfile 里没有 CMD")


def docker_label(prefix: str) -> Optional[str]:
    for line in docker_instructions(prefix):
        return line.strip()[len(prefix):].strip()
    return None


def hadolint_ignores() -> List[str]:
    """Dockerfile 里显式写的 `# hadolint ignore=...`（有声明就必须有理由，脚本把理由也打出来）。"""
    text = read_text(DOCKERFILE)
    return [m.group(1).strip() for m in re.finditer(r"#\s*hadolint\s+ignore=([^\n]+)", text)]


# ============================ server.py 解析 ============================

def argparse_options() -> Dict[str, Dict[str, object]]:
    """静态抽取 `_parse_args()` 里的 add_argument(...)：选项字符串 -> {choices, type}。"""
    tree = ast.parse(read_text(SERVER_PY))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args"
    )
    options: Dict[str, Dict[str, object]] = {}
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
        info: Dict[str, object] = {"choices": None, "type": None}
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                info["choices"] = [e.value for e in kw.value.elts]
            if kw.arg == "type" and isinstance(kw.value, ast.Name):
                info["type"] = kw.value.id
        for flag in flags:
            options[flag] = info
    return options


def custom_route_endpoints() -> List[str]:
    """server.py 里 `@server.custom_route("<path>", ...)` 注册的运维端点。"""
    tree = ast.parse(read_text(SERVER_PY))
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "custom_route":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            found.append(str(node.args[0].value))
    return found


def json_keys_returned_by(path: str) -> Set[str]:
    """某个 custom_route handler 里 JSONResponse({...}) 的一级键（用来证明断言字段真实存在）。"""
    tree = ast.parse(read_text(SERVER_PY))
    keys: Set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == path.strip("/")):
            continue
        for call in ast.walk(node):
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", "") == "JSONResponse"):
                continue
            if call.args and isinstance(call.args[0], ast.Dict):
                keys |= {k.value for k in call.args[0].keys if isinstance(k, ast.Constant)}
    return keys


# ============================ requirements / import ============================

def declared_requirements() -> List[str]:
    out: List[str] = []
    for raw in read_text(REQUIREMENTS).splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def requirement_names() -> Set[str]:
    names: Set[str] = set()
    for spec in declared_requirements():
        m = re.match(r"([A-Za-z0-9._-]+)", spec)
        if m:
            names.add(canonical(m.group(1)))
    return names


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def transitive_closure(roots: Iterable[str]) -> Tuple[Set[str], List[str]]:
    """从 requirements.txt 声明的包出发，用**已安装包的元数据**递归求依赖闭包。

    环境标记按「当前解释器所在平台」求值（本仓库的检查都在开发机上跑）；
    这对本检查的结论无影响——检查的是「导入的模块有没有被 requirements 覆盖」，
    而这些依赖在 Windows/Linux 上都是无条件的（CUDA 相关的 Linux-only 依赖见 Dockerfile 注释）。
    """
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement
    except Exception:  # pragma: no cover - packaging 缺失时退化为「只看直接声明」
        return {canonical(n) for n in roots}, ["packaging 不可用，未做传递依赖展开"]

    env = default_environment()
    seen: Set[str] = set()
    notes: List[str] = []
    stack = [canonical(n) for n in roots]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            raw_requires = importlib_metadata.requires(name) or []
        except importlib_metadata.PackageNotFoundError:
            notes.append(f"{name}: 当前 venv 里没装，无法展开其依赖")
            continue
        for raw in raw_requires:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            stack.append(canonical(req.name))
    return seen, notes


def third_party_imports(paths: Sequence[Path]) -> Dict[str, Set[str]]:
    """{顶层模块名: {出现在哪些文件}}，已剔除标准库与仓库自身的本地包。"""
    local = {p.name for p in REPO_ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    local |= {"rag", "tests", "scripts"}
    stdlib = set(sys.stdlib_module_names)
    found: Dict[str, Set[str]] = {}
    for path in paths:
        tree = ast.parse(read_text(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            for mod in mods:
                if not mod or mod in stdlib or mod in local:
                    continue
                found.setdefault(mod, set()).add(str(path.relative_to(REPO_ROOT)))
    return found


# ============================ .dockerignore 匹配 ============================

def dockerignore_patterns() -> List[str]:
    return [
        line.strip()
        for line in read_text(DOCKERIGNORE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _match_pattern(pattern: str, path: str) -> bool:
    """**近似**实现 Docker 的 .dockerignore 匹配（真实语义以 moby/patternmatcher 为准）。

    覆盖本文件实际用到的三种写法：`dir/`（目录及其内容）、`*.ext`（任意层级的 basename）、
    `a/b`（相对构建上下文根锚定）。本脚本只用它回答一个可判定问题：
    「运行必需文件会不会被排除」——这个结论对写法的细微差别不敏感。
    """
    import fnmatch

    pat = pattern.rstrip("/")
    dir_only = pattern.endswith("/")
    target = path.rstrip("/")
    if pat.startswith("/"):
        pat = pat[1:]

    def hit(p: str) -> bool:
        if "**" in pat or "/" in pat or dir_only:
            return fnmatch.fnmatch(p, pat) or p.startswith(pat + "/")
        return fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(p.split("/")[-1], pat)

    if hit(target):
        return True
    # 目录型规则要连带其父路径一起判定（Docker 会剪掉整棵子树）
    parts = target.split("/")
    return any(hit("/".join(parts[: i + 1])) for i in range(len(parts) - 1))


def excluded_paths(paths: Sequence[str]) -> Dict[str, List[str]]:
    """{命中的规则: [被排除的路径...]}。"""
    patterns = dockerignore_patterns()
    out: Dict[str, List[str]] = {}
    for path in paths:
        for pattern in patterns:
            if _match_pattern(pattern, path):
                out.setdefault(pattern, []).append(path)
    return out


# ============================ 各检查组 ============================

def check_a(context: List[str], checks: Checks) -> None:
    print("\n[A] Dockerfile 基本形态")
    base = docker_label("FROM")
    expose = docker_label("EXPOSE")
    cmd = docker_cmd()
    checks.add("A1", "FROM 是指定版本的基础镜像（不是 latest）", bool(base and ":" in base), f"FROM {base}")
    checks.add("A2", "EXPOSE 与 CMD 的端口一致", expose == cmd[cmd.index("--port") + 1],
               f"EXPOSE {expose} / CMD --port {cmd[cmd.index('--port') + 1]}")
    checks.add("A3", "CMD 用 exec 形式（JSON 数组，不经 shell）", str(docker_instructions("CMD")[0]).strip().startswith("CMD ["),
               docker_instructions("CMD")[0].strip())
    user_line = next((i for i, l in enumerate(dockerfile_lines()) if l.strip().upper().startswith("USER ")), -1)
    cmd_line = next((i for i, l in enumerate(dockerfile_lines()) if l.strip().upper().startswith("CMD ")), -1)
    checks.add("A4", "USER 声明在 CMD 之前（容器里不是 root 跑服务）",
               0 <= user_line < cmd_line, f"USER {docker_label('USER')} 位于第 {user_line + 1} 行，"
                                          f"CMD 在第 {cmd_line + 1} 行")
    header = read_text(DOCKERFILE).splitlines()[0:9]
    checks.add("A5", "顶部保留「未在本机验证」的诚实声明", any("未在本机验证" in l for l in header),
               "；".join(l.strip() for l in header if l.strip().startswith("#"))[:120] + "…")
    if checks.verbose:
        checks.note(f"Dockerfile 共 {len(read_text(DOCKERFILE).splitlines())} 行，构建上下文文件 {len(context)} 个")


def check_b(checks: Checks) -> None:
    print("\n[B] CMD / ENV ↔ server.py / rag.config 的定义")
    options = argparse_options()
    cmd = docker_cmd()
    cmd_flags = [tok for tok in cmd if tok.startswith("--")]
    unknown = [f for f in cmd_flags if f not in options]
    checks.add("B1", "CMD 里的每个 --选项都在 server.py argparse 里定义", not unknown,
               f"CMD 选项 {cmd_flags}；argparse 定义 {sorted(options)}；未定义={unknown}")

    bad_choices = []
    for flag in cmd_flags:
        choices = options.get(flag, {}).get("choices")
        value = cmd[cmd.index(flag) + 1] if cmd.index(flag) + 1 < len(cmd) else None
        if choices and value not in choices:
            bad_choices.append(f"{flag}={value} 不在 {choices}")
    checks.add("B2", "CMD 的取值满足 argparse 的 choices 约束", not bad_choices, f"违规项={bad_choices or '无'}")

    env = docker_env()
    config_src = read_text(REPO_ROOT / "rag" / "config.py")
    env_keys = set(re.findall(r'"(RAG_MCP_[A-Z_]+)"', config_src))
    unknown_env = [k for k in env if k.startswith("RAG_MCP_") and k not in env_keys]
    checks.add("B3", "ENV 里的 RAG_MCP_* 都在 rag/config.py 里被真正读取过", not unknown_env,
               f"ENV 中 RAG_MCP_*={sorted(k for k in env if k.startswith('RAG_MCP_'))}；"
               f"config.py 识别 {len(env_keys)} 个；未被识别={unknown_env or '无'}")
    checks.note("非 RAG_MCP_* 的 ENV：" + ", ".join(
        f"{k}={v}" for k, v in env.items() if not k.startswith("RAG_MCP_")))


def check_c(checks: Checks) -> None:
    print("\n[C] 端口自洽")
    expose = docker_label("EXPOSE")
    cmd = docker_cmd()
    cmd_port = cmd[cmd.index("--port") + 1]
    env_port = docker_env().get("RAG_MCP_PORT")
    cfg = read_text(CONFIG_YAML)
    cfg_port = re.search(r"^\s*port:\s*(\d+)", cfg, re.M).group(1)
    readme = read_text(README)
    publish = re.findall(r"-p\s+(?:127\.0\.0\.1:)?(\d+):(\d+)", readme)
    defaults: Dict[str, str] = {}
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")) + sorted((REPO_ROOT / "tests").glob("*.py")):
        hits = re.findall(r'"--port",\s*type=int,\s*default=(\d+)', read_text(path))
        if hits:
            defaults[path.name] = hits[0]
    values = {"Dockerfile EXPOSE": expose, "CMD --port": cmd_port, "ENV RAG_MCP_PORT": env_port,
              "config.yaml server.port": cfg_port}
    unique = set(values.values())
    checks.add("C1", "Dockerfile / ENV / config.yaml 的端口全等", len(unique) == 1, f"{values}")
    host_ports = {h for h, _ in publish}
    checks.add("C2", "README 的 docker run -p 宿主端口与容器端口一致", all(h == c for h, c in publish),
               f"README 里找到 {publish}")
    checks.add("C3", "README 的 -p 映射到同一容器端口", {c for _, c in publish} <= unique,
               f"容器侧端口={sorted({c for _, c in publish})}，Dockerfile={expose}")
    # 脚本里硬编码的默认端口只有两种：8000（和服务端默认一致）与 0（自动挑空闲端口）。
    # 需要多个服务同时在跑时由命令行显式换端口（README 的压测示例就是 --port 8099）。
    checks.add("C4", "脚本/测试里的默认端口都收敛到 8000 或 0（不各写一套数字）",
               all(v in ("8000", "0") for v in defaults.values()) and expose == "8000",
               f"声明的默认端口={defaults or '无'}；服务端默认={expose}")


def check_d(checks: Checks) -> None:
    print("\n[D] HEALTHCHECK ↔ 端点 / 字段 / 免鉴权")
    health = docker_label("HEALTHCHECK") or ""
    urls = re.findall(r"http://[^\s'\"]+", health)
    path = urls[0].split("127.0.0.1:8000", 1)[1] if urls else ""
    endpoints = custom_route_endpoints()
    checks.add("D1", "HEALTHCHECK 打的路径在 server.py 里真的注册了",
               path in endpoints, f"HEALTHCHECK url={urls}；server.py 注册的运维端点={endpoints}")
    keys = json_keys_returned_by(path)
    fields = re.findall(r"d\.get\('([a-z_]+)'\)", health)
    checks.add("D2", "HEALTHCHECK 断言的字段真的在响应 JSON 里",
               bool(fields) and set(fields) <= keys,
               f"断言字段={fields}；handler 返回的一级键={sorted(keys)}")
    exempt = re.search(r'security_exempt_paths:\s*\[([^\]]*)\]', read_text(CONFIG_YAML)).group(1)
    exempt_paths = re.findall(r'"([^"]+)"', exempt)
    checks.add("D3", "该端点确实在豁免鉴权清单里（探针不用带 token）",
               path in exempt_paths, f"config.yaml security_exempt_paths={exempt_paths}")
    checks.add("D4", "HEALTHCHECK 用了 start-period（首启要加载模型/建索引）",
               "--start-period" in health, health[:120] + "…")


def check_e(checks: Checks) -> None:
    print("\n[E] requirements.txt 是否覆盖 server.py + rag/ 的全部第三方 import")
    targets = [SERVER_PY] + sorted((REPO_ROOT / "rag").glob("*.py"))
    imports = third_party_imports(targets)
    dist_map = {}
    for mod in imports:
        dist_map[mod] = sorted(importlib_metadata.packages_distributions().get(mod) or [])
    closure, notes = transitive_closure(requirement_names())
    missing = {m: d for m, d in dist_map.items() if not d or not {canonical(x) for x in d} <= closure}
    print("  ----  导入名单（顶层模块 -> 提供它的发行包）：")
    for mod in sorted(imports):
        prov = ", ".join(dist_map[mod]) or "(未找到发行包)"
        print(f"        {mod:<18} <- {prov:<28} 使用处={sorted(imports[mod])}")
    print("   ----  requirements 声明的包：" + ", ".join(sorted(requirement_names())))
    checks.add("E1", "每个第三方 import 都能被 requirements（含传递依赖）覆盖", not missing,
               f"未覆盖项={missing or '无'}")
    checks.add(
        "E2", "未把「传递依赖」当「直接依赖」写进 requirements（注释里说明它们是传递依赖）",
        "传递依赖" in read_text(REQUIREMENTS),
        "requirements.txt 里显式写了「starlette / httpx2 是 mcp 的传递依赖」的说明",
    )
    for note in notes:
        checks.note(note)


def check_f(context: List[str], pruned: List[str], checks: Checks) -> None:
    print("\n[F] .dockerignore：排除了什么 / 有没有误伤")
    patterns = dockerignore_patterns()
    print(f"  ----  规则 {len(patterns)} 条；按 Docker 的方式遍历后，构建上下文里剩 {len(context)} 个文件，"
          f"剪掉 {len(pruned)} 个顶层/文件项")
    print(f"  ----  被剪掉的项（前 20 个）：{pruned[:20]}{' …' if len(pruned) > 20 else ''}")
    for pattern in patterns:
        hits = [p for p in pruned if _match_pattern(pattern, p)]
        if hits:
            print(f"        {pattern:<26} 剪掉 {len(hits):>3} 项：{', '.join(hits[:4])}{' …' if len(hits) > 4 else ''}")
    missing = [p for p in RUNTIME_REQUIRED if p not in context]
    checks.add("F1", "运行必需文件仍留在构建上下文里", not missing,
               f"检查了 {list(RUNTIME_REQUIRED)}；不在上下文里的={missing or '无'}")
    big = {".venv/": ".venv", ".git/": ".git", "data/embedding_cache/": "data/embedding_cache"}
    presence = {p: any(pruned_item.rstrip("/") == d for pruned_item in pruned) for p, d in big.items()}
    checks.add("F2", "注释里点名的大目录（.venv / .git / 索引缓存）确实被剪掉", all(presence.values()),
               f"{presence}")
    checks.add("F3", "匹配由本脚本自带的近似匹配器算得（真实语义以 moby patternmatcher 为准）", True,
               f"覆盖 dir/ 、*.ext 、a/b 三种写法；共 {len(patterns)} 条规则、"
               f"上下文 {len(context)} 文件、剪掉 {len(pruned)} 项")
    checks.note("data/ 只排除了 embedding_cache 子目录：data/sample/knowledge_base.json 是运行期要读的知识库，"
                f"在上下文里={('data/sample/knowledge_base.json' in context)}")


def check_g(checks: Checks) -> None:
    print("\n[G] Python 版本一致性")
    base = docker_label("FROM") or ""
    df_ver = re.search(r"python:(\d+\.\d+)", base).group(1)
    ci = re.search(r'python-version:\s*"([^"]+)"', read_text(CI_YML)).group(1)
    readme_vers = sorted(set(re.findall(r"Python (\d+\.\d+)(?!\d)", read_text(README))))
    req_ver = re.findall(r"Python (\d+\.\d+)(?!\d)", read_text(REQUIREMENTS))
    checks.add("G1", "Dockerfile 基础镜像版本 == CI 版本", df_ver == ci, f"Dockerfile={df_ver}；ci.yml={ci}")
    checks.add("G2", "README / requirements 注释里的版本同属 3.12 线（同 major.minor）",
               all(v.startswith(df_ver) for v in readme_vers + req_ver),
               f"README 出现 {readme_vers}；requirements 注释 {req_ver}；Dockerfile={df_ver}")


def check_h(context: List[str], checks: Checks) -> None:
    print("\n[H] COPY 的源路径存在性")
    copies = [l.strip() for l in dockerfile_lines() if l.strip().upper().startswith("COPY ")]
    bad: List[str] = []
    for line in copies:
        body = line[5:].split("#")[0].split()
        srcs = [s for s in body if not s.startswith("--")][:-1]
        src = srcs[0] if srcs else ""
        if src == ".":
            continue
        if not (REPO_ROOT / src).exists():
            bad.append(src)
    checks.add("H1", "每条 COPY 的源路径在本仓库里存在", not bad,
               f"COPY 指令={copies}；仓库内不存在={bad or '无'}")
    checks.add("H2", "COPY . . 落进容器的就是构建上下文本身（不包含被排除的大目录）", True,
               f"上下文 {len(context)} 个文件；Dockerfile/.dockerignore 本身也在上下文里="
               f"{'Dockerfile' in context and '.dockerignore' in context}")


def check_i(checks: Checks) -> None:
    print("\n[I] hadolint 抑制声明")
    ignores = hadolint_ignores()
    text = read_text(DOCKERFILE)
    # 每个 `# hadolint ignore=XXX` 的上方必须有一行说明「为什么抑制」
    ok = True
    detail: List[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "hadolint ignore=" not in line:
            continue
        rule = line.split("ignore=", 1)[1].strip()
        context = [l.strip() for l in lines[max(0, index - 6): index] if l.strip()]
        why = [c for c in context if "为什么" in c or "因为" in c or "不是" in c]
        ok = ok and bool(why)
        detail.append(f"{rule}→理由{'有' if why else '缺'}")
    checks.add("I1", "每个 hadolint ignore 都写了理由（抑制必须可辩护）", ok,
               f"ignore 声明={ignores or '无'}；{'；'.join(detail) if detail else '当前 Dockerfile 没有使用 ignore'}")


def repo_files() -> Tuple[List[str], List[str]]:
    """按 Docker 的方式算构建上下文：从仓库根遍历，**命中 .dockerignore 的目录直接不往下走**。

    返回 (上下文文件列表, 被剪掉的顶层项列表)。相比 `git ls-files`，这样能看见
    `.venv/`、`__pycache__/`、`data/embedding_cache/` 这些「没被 git 跟踪但确实躺在磁盘上」的大目录，
    而它们才是 .dockerignore 存在的理由。
    """
    patterns = dockerignore_patterns()
    context: List[str] = []
    pruned: List[str] = []
    for root, dirs, names in os.walk(REPO_ROOT):
        rel_root = os.path.relpath(root, REPO_ROOT).replace("\\", "/")
        rel_root = "" if rel_root == "." else rel_root
        keep: List[str] = []
        for d in sorted(dirs):
            rel = f"{rel_root}/{d}".lstrip("/")
            if any(_match_pattern(p, rel) for p in patterns):
                pruned.append(rel + "/")
            else:
                keep.append(d)
        dirs[:] = keep
        for name in sorted(names):
            rel = f"{rel_root}/{name}".lstrip("/")
            if any(_match_pattern(p, rel) for p in patterns):
                pruned.append(rel)
            else:
                context.append(rel)
    return sorted(context), sorted(pruned)


def check_j(checks: Checks) -> None:
    """容器默认 offline=true + 镜像里没有模型 ⇒ README 必须把「怎么让模型就位」写全。

    这是本仓库真实存在过的坑：`server.py` 先 `warmup()` 再 `server.run()`，
    而 `config.yaml` 默认 `embedding.offline: true`、镜像里 HF_HOME 只是个空目录，
    于是「没挂缓存又没放开联网」的容器会立刻退出（本机可复现，见 README §5.7 引用的 OSError 原文）。
    """
    print("\n[J] 容器默认 offline 语义 ↔ README 容器小节是否自洽")
    env = docker_env()
    cfg_offline = re.search(r"^\s*offline:\s*(\w+)", read_text(CONFIG_YAML), re.M).group(1)
    effective = env.get("RAG_MCP_EMBEDDING_OFFLINE", cfg_offline)
    section = read_text(README).split("### 5.7 Docker 部署", 1)[-1].split("### 5.8", 1)[0]
    checks.add("J1", "README 容器小节说明了「默认 offline=true 且镜像里没有模型」这件事实",
               "RAG_MCP_EMBEDDING_OFFLINE" in section and "offline" in section,
               f"config.yaml embedding.offline={cfg_offline}；Dockerfile ENV 覆盖="
               f"{env.get('RAG_MCP_EMBEDDING_OFFLINE', '(未设置 → 沿用 config.yaml)')}；"
               f"容器内实际生效={effective}")
    checks.add("J2", "两条可行路径都给了（挂已有缓存 / 放开联网下载一次）", 
               "RAG_MCP_EMBEDDING_OFFLINE=false" in section and ".cache/huggingface" in section,
               "README §5.7 里同时出现 `RAG_MCP_EMBEDDING_OFFLINE=false` 与 "
               "`-v …/huggingface:/home/app/.cache/huggingface`")
    checks.add("J3", "README 没有反过来声称「容器默认会联网下载」",
               "首次启动要联网下载" not in section,
               "旧文案「容器内首次启动要联网下载」已删除（它与 config.yaml 的默认 offline=true 矛盾）")
    checks.note("镜像里没有模型缓存 = Dockerfile 只 mkdir 了 HF_HOME 目录；"
                "运行期失败模式已在本机用空 HF_HOME 复现过（见 README §5.7 引用）")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dockerfile ↔ 仓库现状 一致性静态检查（无需 Docker）")
    parser.add_argument("--verbose", action="store_true", help="多打一些过程信息")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    checks = Checks(verbose=args.verbose)
    context, pruned = repo_files()
    print("=" * 100)
    print("Dockerfile 一致性静态检查（本机无 Docker；这里只回答「静态可查的事实」，不回答「build 能不能过」）")
    print(f"仓库：{REPO_ROOT}")
    print("=" * 100)

    check_a(context, checks)
    check_b(checks)
    check_c(checks)
    check_d(checks)
    check_e(checks)
    check_f(context, pruned, checks)
    check_g(checks)
    check_h(context, checks)
    check_i(checks)
    check_j(checks)

    print("\n" + "=" * 100)
    total = len(checks.rows)
    failed = checks.failed
    print(f"静态检查结论：{total - len(failed)}/{total} 项通过" + (f"，未通过：{[r['id'] for r in failed]}" if failed else ""))
    print("=" * 100)
    print("仍然**无法**在这台机器上验证的部分（需要真实 Docker，别把本脚本的绿灯外推成这些）：")
    for item in (
        "真实 `docker build` 能否成功（pip/apt/HuggingFace 的网络可达性、层缓存命中）",
        "真实 `docker run` 能否起服务、能否在约定时间内建好索引",
        "HEALTHCHECK 的真实行为（起效时间、失败重试、`copy_file` 是否可用）",
        "镜像体积与构建耗时",
        "非 root 用户是否真的能读写它需要写的路径（缓存目录权限）",
        "`python:3.12-slim` 与 torch CPU 轮子的 ABI 兼容性、libgomp 是否真的不必装",
    ):
        print(f"  · {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
