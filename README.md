````md
# GPU Parquet Scoring (Docker Compose + NVIDIA + uv)

This project runs the script `rate_st_translation.py`, which:
- loads parallel datasets (Hugging Face Datasets),
- computes metrics in batches on GPU (MEXMA cosine or MetricX QE),
- writes results as sharded Parquet files (e.g., zstd compression),
- supports resuming from existing shards.

Everything runs inside an NVIDIA GPU-enabled container, with dependencies installed via `uv`.
The whole project directory is mounted as a volume, so:
- **Parquet outputs** are written into your project folder on the host,
- **datasets/models caches** are also stored inside your project folder (e.g. `./.cache/...`).

---

## Host requirements

1. NVIDIA drivers working (`nvidia-smi` works on the host).
2. Docker + Docker Compose installed.
3. **nvidia-container-toolkit** installed (so containers can access the GPU).

Quick GPU test:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
````

---

## Repository files

* `Dockerfile` — NVIDIA CUDA base image + Python + `uv`, venv in `/opt/venv`
* `docker-compose.yml` — `scorer` service with `gpus: all` and `.:/workspace` volume mount
* `requirements.txt` — Python dependencies installed via `uv pip install -r requirements.txt`
* `rate_st_translation.py` — the main script

---

## Build the image

From the project directory:

```bash
docker compose build
```

---

## Running

### 1) Short test run with 1024 samples only
Choose the appropriate model name:
```bash
docker compose run --rm scorer \
  python rate_st_translation.py \
    --test_run \
    --model_name google/metricx-24-hybrid-xl-v2p6-bfloat16 \
    --metricx_bf16 \
    --encode_batch_size 1 \
    --subset en-pl \
    --max_length 512
```
For multi-gpu (choose appropriate values for `devices` and `num_workers`):
```bash
docker compose run --rm scorer \
  python rate_st_translation_multi_gpu.py \
    --model_name google/metricx-24-hybrid-large-v2p6-bfloat16 \
    --subset en-pl \
    --scorer metricx_qe \
    --devices 0,1,2,3 \
    --num_workers 4 \
    --test_run \
    --test_samples 1024 \
    --batch_size 64 \
    --encode_batch_size 16 \
    --queue_max_batches 16
```

### 2) MEXMA (cosine similarity of embeddings)

```bash
docker compose run --rm scorer \
  python rate_st_translation.py --model_name facebook/MEXMA --subset en-pl
```

### 3) MetricX QE Large (reference-free, prediction 0..25, lower is better)
Important - choose the appropriate `--encode_batch_size`
```bash
docker compose run --rm scorer \
  python rate_st_translation.py \
    --model_name google/metricx-24-hybrid-large-v2p6 \
    --encode_batch_size 16 \
    --subset en-pl \
    --max_length 0
```

### 4) MetricX QE XL BFP16 (reference-free, prediction 0..25, lower is better)
Important - choose the appropriate `--encode_batch_size`
```bash
docker compose run --rm scorer \
  python rate_st_translation.py \
    --model_name google/metricx-24-hybrid-xl-v2p6-bfloat16 \
    --metricx_bf16 \
    --encode_batch_size 8 \
    --subset en-pl \
    --max_length 0
```

For multi-gpu (choose appropriate values for `devices` and `num_workers`):
```bash
docker compose run --rm scorer \
  python rate_st_translation_multi_gpu.py \
    --model_name google/metricx-24-hybrid-xl-v2p6-bfloat16 \
    --metricx_bf16 \
    --encode_batch_size 8 \
    --subset en-pl \
    --max_length 0 \
    --batch_size 1024 \
    --queue_max_batches 8 \
    --devices 0,1,2,3 \
    --num_workers 4
```


### 5) MetricX QE XXL BFP16 (reference-free, prediction 0..25, lower is better)
Important - choose the appropriate `--encode_batch_size`:
```bash
docker compose run --rm scorer \
  python rate_st_translation.py \
    --model_name google/metricx-24-hybrid-xxl-v2p6-bfloat16 \
    --metricx_bf16 \
    --encode_batch_size 4 \
    --subset en-pl \
    --max_length 0
```

---

## Where data is stored

### Output

By default the script writes to:

* `./outputs_<model>_parquet/...` (inside the project directory on the host)

### Caches (datasets/models)

`docker-compose.yml` sets environment variables to keep all caches inside the project:

* `./.cache/huggingface/datasets`
* `./.cache/huggingface/hub`
* `./.cache/huggingface/transformers`
* `./.cache/torch`
* `./.cache/uv`

This ensures downloaded datasets and models are **physically stored on the host**, in your project folder.

---

## Resume behavior

The script scans existing shards in the target split directory and continues from the last `end_idx`.
Temporary files use the `.tmp` suffix and are not considered completed shards.
