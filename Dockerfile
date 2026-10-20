# rag-mcp-server 容器镜像（HTTP 形态）
#
# ★★ 未在本机验证 ★★
#   开发机（Windows）上没有安装 Docker，本文件没有跑过 `docker build` / `docker run`，
#   是按官方最佳实践写的「待验证」产物。请按下面命令自行验证（README §5.7 有同样说明）：
#       docker build -t rag-mcp-server:0.1.0 .
#       docker run --rm -p 127.0.0.1:8000:8000 -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0
#       curl http://127.0.0.1:8000/healthz
#
#   「未验证」到什么程度，划清楚（本文件能静态取证的都取证了，动态的一件没验）：
#   - 已静态取证（可重复跑：.venv\Scripts\python.exe scripts/check_dockerfile_consistency.py，
#     29 项全过；另有 hadolint 2.12.0 静态检查 0 告警）：
#       · CMD 的每个选项/取值都真在 server.py 的 argparse 里（不是照着文档抄的）；
#       · ENV 里的 RAG_MCP_* 都真被 rag/config.py 读取，且与 config.yaml 的端口一致；
#       · /healthz 真的用 @server.custom_route 注册、返回体真的有一级字段 index_ready、确实在
#         security_exempt_paths 里（免鉴权，所以探针不需要 token）；
#       · server.py + rag/ 的全部第三方 import 都被 requirements.txt（含传递依赖）覆盖；
#       · 基础镜像 layer 清单里确实有 usr/sbin/useradd 与 usr/sbin/nologin（本层能跑）；
#       · COPY 的源路径都存在；.dockerignore 没误伤运行必需文件。
#   - 仍未验证（没有 Docker 就回答不了，别把上面的绿灯外推）：build 能否成功、run 能否起服务、
#     HEALTHCHECK 的真实行为、镜像体积、非 root 权限是否真的够用、apt/pip/HuggingFace 的网络可达性。
#
# 关于模型：config.yaml 默认 `embedding.offline: true`（只读本地 HuggingFace 缓存）。
#   本镜像里**没有**缓存（HF_HOME 只是个空目录），所以二选一，且必须显式选：
#     1) 挂载宿主机上已经下好的缓存（推荐，构建/启动都离线、可复现）：
#        docker run --rm -p 8000:8000 \
#          -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface:ro" \
#          -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0
#     2) 首次运行允许联网下载（BGE-small-zh-v1.5 约 95MB），缓存落到卷里留给下次：
#        docker run --rm -p 8000:8000 -v rag-hf-cache:/home/app/.cache/huggingface \
#          -e RAG_MCP_EMBEDDING_OFFLINE=false \
#          -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0
#   ★ 别漏掉第 2 条里的开关：本机实测（把 HF_HOME 指到空目录、不设 RAG_MCP_EMBEDDING_OFFLINE）warmup 直接抛
#     `OSError: ... couldn't find them in the cached files`，而 server.py 是**先 warmup 再 server.run()**，
#     于是容器立刻退出——「没挂缓存又不放开联网」是起不来，不是起得慢。

FROM python:3.12-slim

# PYTHONUNBUFFERED：容器日志要立刻可见，否则被缓冲住、出问题看不到
# PIP_NO_CACHE_DIR：pip 缓存不留在镜像层里，镜像更小
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    RAG_MCP_TRANSPORT=streamable-http \
    RAG_MCP_HOST=0.0.0.0 \
    RAG_MCP_PORT=8000

# libgomp1（OpenMP 运行时）。取证过的事实，别照抄印象：
#   · 基础镜像里**确实没有** libgomp.so.1：拉 python:3.12-slim 的 4 个 layer（压缩 43.3MB、5660 个 tar 条目）
#     按文件名扫 libgomp → 命中 0；
#   · 但 pip 装下来的轮子**自带**一份：torch 的 manylinux 轮子里有 `torch/lib/libgomp.so.1`，
#     scikit-learn 的轮子里有 `scikit_learn.libs/libgomp-e985bcbb.so.1.0.0`。
# 所以这一层是**防御性**的（约 150KB），不是 torch 的硬依赖：留着的理由是——任何按裸名字
# dlopen("libgomp.so.1")（不带 RPATH）的扩展在缺它会 ImportError，而本机没有 Docker，无法用真实
# build/run 证伪这条风险。要证伪的机器删掉本层即可（前提：一起删掉下面的 hadolint ignore）。
#
# 为什么用 ignore 而不是按 DL3008 把包钉成 `libgomp1=<bookworm 当前点版本>`：Debian 仓库只保留当前一版
# （bookworm-updates 会轮换），钉死点版本号 = 下次仓库轮换后 apt-get install 直接找不到包、构建失败；
# 真要说「钉」，该钉的是基础镜像自身（FROM python:3.12-slim@sha256:…），但那会牺牲多架构可构建性，
# 本文件按取舍选择不钉。DL3008 这条告警在本文件里是**已知且刻意**保留的，理由即上述。
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先把 torch 落在**官方 CPU 索引**上，理由都取证过：
#   · PyPI 上的 torch 2.14.1 在 Linux 侧装到的是 554.6MB 的 **CUDA** 轮子（内含 libtorch_cuda.so），
#     其 METADATA 还要求 `cuda-toolkit[cublas,…]==13.0.3`、`nvidia-cudnn-cu13`、`nvidia-nccl-cu13`、
#     `nvidia-cusparselt-cu13`、`nvidia-nvshmem-cu13`、`triton~=3.8.0`（都是 platform_system=="Linux" 的条件依赖）
#     ——本镜像既不挂 GPU 也没有 nvidia runtime，这几个 GB 全是死重；
#   · requirements.txt 里「PyPI 上的 torch 为 CPU 版」那句只在 Windows 侧成立，Linux 侧不成立；
#   · 官方 CPU 索引上有 `torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`（196.3MB，无任何 CUDA 依赖）。
# 为什么下一条 `pip install -r requirements.txt` 不会把它再换回 CUDA 版：PEP 440 里 `==2.14.1` 接受本地版本号
#   `2.14.1+cpu`（本机用 packaging 实跑验证：Version('2.14.1+cpu') in SpecifierSet('==2.14.1') → True），
#   所以 pip 看到 torch 已满足就直接跳过，不会重新下载。
# 为什么用 --extra-index-url 而不是 --index-url：主索引仍是 PyPI（filelock/sympy/jinja2 等依赖只在 PyPI），
#   只给 CPU 索引会解析不到它们；同时把版本写死成 `==2.14.1+cpu`，堵掉「多索引同名包」的混淆空间。
# 这一层刻意放在 `COPY requirements.txt` **之前**：torch 与依赖清单互不影响，
#   这样 requirements.txt 每次改动都不会连带重下这 196MB（也顺便避开 hadolint DL3059「连续 RUN」）。
# 若目标网络访问不了 download.pytorch.org：删掉这一行即可退回 PyPI 的 CUDA 轮子（功能一样，只是镜像大几个 GB）。
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"

# 再拷依赖清单：依赖没变时这一层能命中构建缓存，改代码不必重装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码（.dockerignore 已排除 .venv / .git / 索引缓存等大目录）
# 「已排除」是核对过的，不是照着注释信：按 Docker 的规则遍历仓库后，构建上下文只剩 29 个文件，
# 被剪掉的是 .venv/（本机 1046MB）、.git/、data/embedding_cache/、__pycache__/、*.pyc、.pytest_cache/ 等。
COPY . .

# 非 root 运行：容器里跑服务的进程不该有 root 权限
# useradd 与 /usr/sbin/nologin 都来自基础镜像自带（已在 layer 文件清单里核对到 usr/sbin/useradd、usr/sbin/nologin）
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && mkdir -p /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app/.cache
USER app

EXPOSE 8000

# 健康检查打 /healthz（该端点不鉴权，所以探针里不需要放 token）：
# 已取证：server.py 用 @server.custom_route("/healthz") 注册、响应 JSON 的一级字段里有 index_ready、
# 且 /healthz 在 config.yaml 的 server.security_exempt_paths 里（免鉴权、也免限流，探针不会被 429 误杀）；
# start-period 给 120s——首次启动要加载模型并建索引（本机 CPU 实测就绪约 20s，容器里只会更慢）
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)); sys.exit(0 if d.get('index_ready') else 1)"

# 容器里跑的是 HTTP 形态：显式绑定 0.0.0.0（对外暴露由端口映射 -p 收口），
# 默认 stdio 是给客户端拉起子进程用的，不适合容器
CMD ["python", "server.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
