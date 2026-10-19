# rag-mcp-server 容器镜像（HTTP 形态）
#
# ★★ 未在本机验证 ★★
#   开发机（Windows）上没有安装 Docker，本文件没有跑过 `docker build` / `docker run`，
#   是按官方最佳实践写的「待验证」产物。请按下面命令自行验证（README §5.7 有同样说明）：
#       docker build -t rag-mcp-server:0.1.0 .
#       docker run --rm -p 127.0.0.1:8000:8000 -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0
#       curl http://127.0.0.1:8000/healthz
#
# 关于模型：config.yaml 默认 `embedding.offline: true`（只读本地 HuggingFace 缓存）。
#   容器里没有缓存，所以二选一：
#     1) 挂载宿主机的模型缓存（推荐，构建/启动都离线、可复现）：
#        docker run --rm -p 8000:8000 \
#          -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface:ro" \
#          -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0
#     2) 首次运行允许联网下载（约 95MB），下载完成后缓存留在容器里：
#        docker run --rm -p 8000:8000 -e RAG_MCP_EMBEDDING_OFFLINE=false \
#          -e RAG_MCP_AUTH_TOKEN=change-me rag-mcp-server:0.1.0

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

# libgomp1：PyTorch CPU 轮子的 OpenMP 运行时依赖，debian slim 基础镜像里没有
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只拷依赖清单：依赖没变时这一层能命中构建缓存，改代码不必重装 torch
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷代码（.dockerignore 已排除 .venv / .git / 索引缓存等大目录）
COPY . .

# 非 root 运行：容器里跑服务的进程不该有 root 权限
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
 && mkdir -p /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app/.cache
USER app

EXPOSE 8000

# 健康检查打 /healthz（该端点不鉴权，所以探针里不需要放 token）：
# start-period 给 120s——首次启动要加载模型并建索引（CPU 上约 5~30s），这期间的失败不算数
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD python -c "import json,sys,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)); sys.exit(0 if d.get('index_ready') else 1)"

# 容器里跑的是 HTTP 形态：显式绑定 0.0.0.0（对外暴露由端口映射 -p 收口），
# 默认 stdio 是给客户端拉起子进程用的，不适合容器
CMD ["python", "server.py", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000"]
