FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

ARG PYTHON_VERSION=3.11.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
  && ln -s /root/.local/bin/uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=automatic
RUN uv python install ${PYTHON_VERSION}

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./

RUN uv venv /opt/venv --python ${PYTHON_VERSION} \
 && uv pip install --no-cache-dir -r requirements.txt

CMD ["python", "score_parallel_parquet.py", "--help"]
