FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

WORKDIR /workspace/weak_mean_flow

# ライブラリのインストールに必要なファイル
COPY requirements.txt constraints-docker.txt /opt/weakflow/

# CUDA 12.1用のPyTorch・torchvisionを明示する
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.5.1" \
    "torchvision==0.20.1"

# 今回の実験に必要な追加ライブラリ
RUN python -m pip install --no-cache-dir \
    -c /opt/weakflow/constraints-docker.txt \
    -r /opt/weakflow/requirements.txt \
    && python -m pip check

# ビルド時にはGPUを使わず、importとバージョンを確認する
RUN python -c \
    "import torch, torchvision; \
    assert torch.__version__.split('+')[0] == '2.5.1'; \
    assert torchvision.__version__.split('+')[0] == '0.20.1'; \
    assert torch.version.cuda == '12.1'; \
    print('torch:', torch.__version__); \
    print('torchvision:', torchvision.__version__)"

# 実際に導入された依存関係も記録する
RUN python -m pip freeze > /opt/weakflow/environment.txt

# コンテナを起動した状態で維持する
CMD ["sleep", "infinity"]
