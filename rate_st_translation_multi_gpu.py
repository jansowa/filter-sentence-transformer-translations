#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharded Parquet writer for (english, polish, metrics...) with resume support.

Supports:
- facebook/MEXMA (cosine similarity of embeddings) [single-process]
- google/metricx-*-* in QE mode (reference-free) [single- or multi-worker CUDA]

Multi-worker MetricX design (works even on 1 GPU):
- main process iterates dataset sequentially (from resume_idx) and enqueues batches
- N worker processes (optionally > number of GPUs), each loads its own MetricX model on an assigned device
  (can be the same device, e.g. multiple workers on cuda:0 for testing)
- results may come out-of-order; main process reorders by start_idx and ONLY writes
  contiguous batches to preserve "resume by max_end shard" correctness

Output layout:
  out_dir/
    <dataset>/
      <subset>/
        <split>/
          shard_000123_000000120000_000000140000.parquet
          ...

Each shard contains:
  idx (int64), english (string), polish (string), <metrics...> (float32)

Examples:
  # Single worker MetricX:
  python rate_st_translation_multi_gpu.py --model_name google/metricx-24-hybrid-large-v2p6 --subset en-pl

  # Force 2 workers on a single GPU (cuda:0) to test multi-worker pipeline:
  python rate_st_translation_multi_gpu.py --model_name google/metricx-24-hybrid-large-v2p6 --subset en-pl --devices 0 --num_workers 2 --test_run

  # Multi-GPU:
  python rate_st_translation_multi_gpu.py --model_name google/metricx-24-hybrid-large-v2p6 --subset en-pl --devices 0,1,2,3
"""

import os
import re
import json
import time
import queue
import signal
import logging
import threading
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset

import pyarrow as pa
import pyarrow.parquet as pq


DATASETS_TO_USE = [
    "sentence-transformers/parallel-sentences-global-voices",  # 45,4k
    "sentence-transformers/parallel-sentences-europarl",  # 614k
    "sentence-transformers/parallel-sentences-opensubtitles",  # 1.06M
    "sentence-transformers/parallel-sentences-jw300",  # 1.27M
    "sentence-transformers/parallel-sentences-talks",  # 293K
    "sentence-transformers/parallel-sentences-tatoeba",  # 56K
    "sentence-transformers/parallel-sentences-wikimatrix",  # 117K
    "sentence-transformers/parallel-sentences-ccmatrix", #74,1M
]

DATASET_MAX_SAMPLES = {
    # "sentence-transformers/parallel-sentences-ccmatrix": 25_400_000,
}

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("score-parquet-multi-worker")


def safe_name(s: str) -> str:
    return s.replace("/", "__").replace(":", "__")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_rename(src: Path, dst: Path) -> None:
    os.replace(str(src), str(dst))


SHARD_RE = re.compile(r"^shard_(\d{6})_(\d{12})_(\d{12})\.parquet$")


def find_resume_state(out_split_dir: Path) -> Tuple[int, int]:
    """
    Return (next_idx, next_shard_id) based on existing shards.
    Shards are considered complete only if they have the .parquet suffix (not .tmp).
    """
    if not out_split_dir.exists():
        return 0, 0

    max_end = 0
    max_shard = -1

    for p in out_split_dir.iterdir():
        if not p.is_file():
            continue
        m = SHARD_RE.match(p.name)
        if not m:
            continue
        shard_id = int(m.group(1))
        start_idx = int(m.group(2))
        end_idx = int(m.group(3))  # end exclusive
        if end_idx > start_idx:
            if end_idx > max_end:
                max_end = end_idx
            if shard_id > max_shard:
                max_shard = shard_id

    return max_end, (max_shard + 1)


def choose_parquet_compression(preferred: str = "zstd") -> str:
    try:
        if pa.Codec.is_available(preferred):
            return preferred
    except Exception:
        pass
    for c in ["zstd", "snappy", "gzip"]:
        try:
            if pa.Codec.is_available(c):
                if c != preferred:
                    logger.warning("Codec '%s' not available. Using '%s'.", preferred, c)
                return c
        except Exception:
            continue
    logger.warning("No common Parquet codecs available? Writing without compression.")
    return "NONE"


@dataclass
class BatchResult:
    start_idx: int
    end_idx: int
    idx: np.ndarray                 # int64 [B]
    english: List[str]              # [B]
    polish: List[str]               # [B]
    metrics: Dict[str, np.ndarray]  # {name: float32 [B]}


class ParquetShardWriter:
    def __init__(
        self,
        out_split_dir: Path,
        q: "queue.Queue[Optional[BatchResult]]",
        stop_event: threading.Event,
        shard_rows: int,
        compression: str,
        metric_names: List[str],
        row_group_size: int = 65536,
    ):
        self.out_split_dir = out_split_dir
        self.q = q
        self.stop_event = stop_event
        self.shard_rows = int(shard_rows)
        self.compression = compression
        self.row_group_size = int(row_group_size)
        self.metric_names = list(metric_names)

        self._buf_idx: List[np.ndarray] = []
        self._buf_en: List[List[str]] = []
        self._buf_pl: List[List[str]] = []
        self._buf_metrics: Dict[str, List[np.ndarray]] = {m: [] for m in self.metric_names}
        self._buf_n = 0

        self.next_idx, self.next_shard_id = find_resume_state(out_split_dir)

    def _flush_to_parquet(self) -> None:
        if self._buf_n == 0:
            return

        ensure_dir(self.out_split_dir)

        idx = np.concatenate(self._buf_idx, axis=0).astype(np.int64, copy=False)
        english: List[str] = [s for chunk in self._buf_en for s in chunk]
        polish: List[str] = [s for chunk in self._buf_pl for s in chunk]

        cols: Dict[str, Any] = {
            "idx": pa.array(idx, type=pa.int64()),
            "english": pa.array(english, type=pa.string()),
            "polish": pa.array(polish, type=pa.string()),
        }

        for m in self.metric_names:
            arr = np.concatenate(self._buf_metrics[m], axis=0).astype(np.float32, copy=False)
            cols[m] = pa.array(arr, type=pa.float32())

        n = len(idx)
        assert len(english) == len(polish) == n
        for m in self.metric_names:
            assert len(cols[m]) == n

        start_idx = int(idx[0])
        end_idx = int(idx[-1]) + 1  # end exclusive

        shard_name = f"shard_{self.next_shard_id:06d}_{start_idx:012d}_{end_idx:012d}.parquet"
        tmp_path = self.out_split_dir / (shard_name + ".tmp")
        final_path = self.out_split_dir / shard_name

        table = pa.table(cols)

        pq.write_table(
            table,
            where=str(tmp_path),
            compression=self.compression if self.compression != "NONE" else None,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=self.row_group_size,
        )

        atomic_rename(tmp_path, final_path)

        self.next_shard_id += 1
        self.next_idx = end_idx

        self._buf_idx.clear()
        self._buf_en.clear()
        self._buf_pl.clear()
        for m in self.metric_names:
            self._buf_metrics[m].clear()
        self._buf_n = 0

    def run(self) -> None:
        logger.info(
            "Writer resume state: next_idx=%d, next_shard_id=%d, dir=%s",
            self.next_idx, self.next_shard_id, str(self.out_split_dir)
        )

        while True:
            item = self.q.get()
            if item is None:
                try:
                    self._flush_to_parquet()
                finally:
                    self.q.task_done()
                break

            self._buf_idx.append(item.idx)
            self._buf_en.append(item.english)
            self._buf_pl.append(item.polish)

            for m in self.metric_names:
                if m not in item.metrics:
                    raise KeyError(f"BatchResult is missing metric '{m}'. Available: {list(item.metrics.keys())}")
                self._buf_metrics[m].append(item.metrics[m])

            self._buf_n += len(item.english)

            if self._buf_n >= self.shard_rows:
                self._flush_to_parquet()

            self.q.task_done()

            if self.stop_event.is_set():
                continue


class BaseScorer:
    metric_names: List[str]

    def score_batch(self, english: List[str], polish: List[str]) -> Dict[str, np.ndarray]:
        raise NotImplementedError


def cosine_sim_from_normalized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


class MexmaCosineScorer(BaseScorer):
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        encode_batch_size: int,
        num_encode_workers: int,
        use_fp16: bool = True,
    ):
        from sentence_transformers import SentenceTransformer

        self.metric_names = ["cosine"]
        self.encode_batch_size = int(encode_batch_size)
        self.num_encode_workers = int(num_encode_workers)

        logger.info("Loading SentenceTransformer: %s", model_name)
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.max_seq_length = int(max_length)
        self.model.eval()

        if use_fp16 and device.startswith("cuda"):
            try:
                self.model = self.model.half()
            except Exception:
                pass

    def score_batch(self, english: List[str], polish: List[str]) -> Dict[str, np.ndarray]:
        texts = english + polish
        emb = self.model.encode(
            texts,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            num_workers=self.num_encode_workers,
        )

        B = len(english)
        en_emb = emb[:B].astype(np.float32, copy=False)
        pl_emb = emb[B:].astype(np.float32, copy=False)
        sims = cosine_sim_from_normalized(en_emb, pl_emb).astype(np.float32, copy=False)
        return {"cosine": sims}


class MetricXQEScorer(BaseScorer):
    """
    Minimal MetricX (MT5ForRegression) implementation + input formatting compatible with MetricX.
    QE: source=EN, hypothesis=PL, reference="" (for MetricX-24-hybrid) or no reference (for MetricX-23-QE).
    """

    EXTRA_ID_10_TOKEN_ID = 250089  # <extra_id_10>

    def __init__(
        self,
        model_name_or_path: str,
        device: str,
        batch_size: int,
        tokenizer_name: str = "google/mt5-xl",
        force_qe: bool = True,
        max_input_length: int = 0,
        fp16: bool = True,
        bf16: bool = False,
    ):
        import copy
        import torch.nn as nn
        from dataclasses import dataclass
        from transformers import AutoTokenizer
        from transformers.modeling_outputs import ModelOutput, BaseModelOutput
        from transformers.models.mt5.modeling_mt5 import MT5Config, MT5PreTrainedModel, MT5Stack

        self.metric_names = ["metricx_pred"]
        self.cfg_batch_size = int(batch_size)
        self.model_name_or_path = model_name_or_path
        self.force_qe = bool(force_qe)

        is_mx24 = "metricx-24" in model_name_or_path.lower()
        is_mx23 = "metricx-23" in model_name_or_path.lower()
        if not (is_mx24 or is_mx23):
            raise ValueError(f"Unrecognized MetricX version in name: {model_name_or_path}")

        if max_input_length and max_input_length > 0:
            self.max_length = int(max_input_length)
        else:
            self.max_length = 1536 if is_mx24 else 1024

        if is_mx24:
            self.prefix_hyp = " candidate: "
            self.prefix_ref = " reference: "
            self.prefix_src = "source: "
        else:
            self.prefix_hyp = "candidate: "
            self.prefix_ref = " reference: "
            self.prefix_src = " source: "

        @dataclass
        class MT5ForRegressionOutput(ModelOutput):
            loss: Optional[torch.Tensor] = None
            predictions: Optional[torch.Tensor] = None

        class MT5ForRegression(MT5PreTrainedModel):
            def __init__(self, config: MT5Config):
                super().__init__(config)
                self.model_dim = config.d_model

                self.shared = nn.Embedding(config.vocab_size, config.d_model)

                encoder_config = copy.deepcopy(config)
                encoder_config.is_decoder = False
                encoder_config.use_cache = False
                encoder_config.is_encoder_decoder = False
                self.encoder = MT5Stack(encoder_config, self.shared)

                decoder_config = copy.deepcopy(config)
                decoder_config.is_decoder = True
                decoder_config.is_encoder_decoder = True
                decoder_config.num_layers = config.num_decoder_layers
                self.decoder = MT5Stack(decoder_config, self.shared)

                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

                self.post_init()

            def forward(
                self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                encoder_outputs: Optional[tuple] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.FloatTensor] = None,
                return_dict: Optional[bool] = True,
                **kwargs,
            ):
                return_dict = True if return_dict is None else return_dict

                if encoder_outputs is None:
                    encoder_outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        return_dict=return_dict,
                    )
                elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=encoder_outputs[0],
                        hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                        attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                    )

                hidden_states = encoder_outputs[0]
                batch_size = input_ids.size(0)
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    0,
                    dtype=torch.long,
                    device=input_ids.device,
                )

                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=hidden_states,
                    encoder_attention_mask=attention_mask,
                    return_dict=return_dict,
                )

                sequence_output = decoder_outputs[0]
                lm_logits = self.lm_head(sequence_output)

                predictions = lm_logits[:, 0, MetricXQEScorer.EXTRA_ID_10_TOKEN_ID]
                predictions = torch.clamp(predictions, 0, 25)

                loss = None
                if labels is not None:
                    loss_fct = nn.MSELoss()
                    labels = labels.to(predictions.device)
                    loss = loss_fct(predictions.view(-1), labels.view(-1))

                return MT5ForRegressionOutput(loss=loss, predictions=predictions)

        logger.info("Loading MetricX model: %s", model_name_or_path)
        self.scorer = MT5ForRegression.from_pretrained(model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            legacy=False,
            use_fast=False,
        )

        self.scorer.eval()
        for p in self.scorer.parameters():
            p.requires_grad = False

        self._device = torch.device(device)
        if device.startswith("cuda") and torch.cuda.is_available():
            self.scorer = self.scorer.to(self._device)
            if bf16:
                self.scorer = self.scorer.bfloat16()
            elif fp16:
                try:
                    self.scorer = self.scorer.half()
                except Exception:
                    pass
        else:
            self.scorer = self.scorer.to(torch.device("cpu"))
            self._device = torch.device("cpu")

        self._is_mx24 = is_mx24
        self._is_mx23 = is_mx23
        self._is_mx23_qe = ("-qe-" in model_name_or_path.lower())

    def _encode(self, prefix: str, text: str) -> List[int]:
        return self.tokenizer.encode(prefix + text, add_special_tokens=False)

    def _build_input_ids(self, source: str, hypothesis: str, reference: Optional[str]) -> List[int]:
        hyp_ids = self._encode(self.prefix_hyp, hypothesis)

        if self._is_mx24:
            src_ids = self._encode(self.prefix_src, source)
            input_ids = src_ids + hyp_ids
            if reference is not None:
                ref_ids = self._encode(self.prefix_ref, reference)
                input_ids += ref_ids
            return input_ids

        if self._is_mx23_qe:
            src_ids = self._encode(self.prefix_src, source)
            return hyp_ids + src_ids

        if reference is None:
            raise ValueError("MetricX-23 (non-QE) requires a reference. Use a -qe- model or MetricX-24-hybrid.")
        ref_ids = self._encode(self.prefix_ref, reference)
        return hyp_ids + ref_ids

    def _collate(self, batch_input_ids: List[List[int]]) -> Dict[str, torch.Tensor]:
        trimmed = [ids[: self.max_length] for ids in batch_input_ids]
        batch = {"input_ids": trimmed}
        enc = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        return enc

    @torch.inference_mode()
    def score_batch(self, english: List[str], polish: List[str]) -> Dict[str, np.ndarray]:
        B = len(english)
        assert B == len(polish)

        if self._is_mx24:
            reference = "" if self.force_qe else ""
        else:
            reference = None

        inputs = [self._build_input_ids(en, pl, reference) for en, pl in zip(english, polish)]

        preds: List[torch.Tensor] = []
        for i in range(0, len(inputs), self.cfg_batch_size):
            batch_ids = inputs[i: i + self.cfg_batch_size]
            batch = self._collate(batch_ids)
            batch = {k: v.to(self._device) for k, v in batch.items()}
            out = self.scorer(**batch)
            preds.append(out.predictions.detach().float().cpu())

        pred = torch.cat(preds, dim=0).view(B).numpy().astype(np.float32, copy=False)
        return {"metricx_pred": pred}


def iter_dataset_slices(
    ds,
    start_idx: int,
    batch_size: int,
    max_samples: Optional[int],
) -> Iterable[Tuple[int, int, Dict[str, Any]]]:
    total = len(ds)
    end_total = min(total, start_idx + max_samples) if max_samples is not None else total

    i = start_idx
    while i < end_total:
        j = min(i + batch_size, end_total)
        batch = ds[i:j]
        yield i, j, batch
        i = j


# -------------------- MULTI-WORKER (MetricX) worker & orchestrator --------------------

TaskPayload = Dict[str, Any]
ResultPayload = Tuple[str, Any]  # ("result", BatchResult) | ("done", rank) | ("error", (...))


def _metricx_worker(
    rank: int,
    device_str: str,
    model_name: str,
    tokenizer_name: str,
    encode_batch_size: int,
    max_input_length: int,
    fp16: bool,
    bf16: bool,
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    stop_ev: "mp.Event",
):
    """
    One process per worker. Can be multiple workers per same GPU (for testing).
    """
    try:
        if device_str.startswith("cuda"):
            # Explicitly set device for this process
            torch.cuda.set_device(int(device_str.split(":")[1]))

        scorer = MetricXQEScorer(
            model_name_or_path=model_name,
            device=device_str,
            batch_size=encode_batch_size,
            tokenizer_name=tokenizer_name,
            force_qe=True,
            max_input_length=max_input_length,
            fp16=fp16,
            bf16=bf16,
        )

        while True:
            task = in_q.get()
            if task is None:
                break

            if stop_ev.is_set():
                # Finish current task anyway; stop flag primarily prevents enqueue in main.
                pass

            start_idx = int(task["start_idx"])
            end_idx = int(task["end_idx"])
            english = task["english"]
            polish = task["polish"]

            metrics = scorer.score_batch(english, polish)
            for k, v in metrics.items():
                metrics[k] = np.asarray(v, dtype=np.float32)

            idx = np.arange(start_idx, end_idx, dtype=np.int64)

            br = BatchResult(
                start_idx=start_idx,
                end_idx=end_idx,
                idx=idx,
                english=english,
                polish=polish,
                metrics=metrics,
            )
            out_q.put(("result", br))

        out_q.put(("done", rank))

    except Exception as e:
        tb = traceback.format_exc()
        out_q.put(("error", (rank, str(e), tb)))


def _parse_devices(args) -> List[str]:
    """
    Returns list of device strings, e.g. ["cuda:0", "cuda:1"] or ["cpu"].
    """
    if not torch.cuda.is_available():
        return ["cpu"]

    if args.devices.strip():
        ids = []
        for part in args.devices.split(","):
            part = part.strip()
            if not part:
                continue
            ids.append(int(part))
        if not ids:
            return ["cuda:0"]
        return [f"cuda:{i}" for i in ids]

    if args.num_gpus == -1:
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)]

    n = max(1, int(args.num_gpus))
    n = min(n, torch.cuda.device_count())
    return [f"cuda:{i}" for i in range(n)]


def _expand_devices_for_workers(devices: List[str], num_workers: int) -> List[str]:
    """
    If num_workers > len(devices), expand by repeating devices round-robin.
    This enables forcing multiple workers on the same GPU (e.g., cuda:0, cuda:0).
    """
    if num_workers <= 0:
        return devices
    if len(devices) == 0:
        return devices
    if num_workers == len(devices):
        return devices
    expanded = []
    for i in range(num_workers):
        expanded.append(devices[i % len(devices)])
    return expanded


# -------------------- MAIN --------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="facebook/MEXMA")
    parser.add_argument(
        "--scorer",
        type=str,
        default="auto",
        choices=["auto", "mexma_cosine", "metricx_qe"],
        help="auto: infer from model name; mexma_cosine: cosine on embeddings; metricx_qe: MetricX QE (0..25, lower is better).",
    )

    parser.add_argument("--subset", type=str, default="en-pl")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--src_col", type=str, default="english")
    parser.add_argument("--tgt_col", type=str, default="non_english")

    parser.add_argument("--out_dir", type=str, default="", help="If empty, set to outputs_<model>_parquet.")

    parser.add_argument("--batch_size", type=int, default=2048, help="How many (EN,PL) pairs per outer batch (dataset slicing).")

    parser.add_argument(
        "--encode_batch_size",
        type=int,
        default=64,
        help="Batch size used inside scorer (MEXMA encode() OR MetricX forward()).",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_encode_workers", type=int, default=max(1, (os.cpu_count() or 8) // 2))

    parser.add_argument("--metricx_tokenizer", type=str, default="google/mt5-xl")
    parser.add_argument("--metricx_max_input_length", type=int, default=0, help="0 = auto (1536 for 24, 1024 for 23).")
    parser.add_argument("--metricx_fp16", action="store_true", help="Force fp16 for MetricX (CUDA).")
    parser.add_argument("--metricx_bf16", action="store_true", help="Force bf16 for MetricX (CUDA).")

    # Worker controls
    parser.add_argument("--devices", type=str, default="", help="Comma-separated CUDA device ids, e.g. '0,1,2,3'. Overrides --num_gpus.")
    parser.add_argument("--num_gpus", type=int, default=1, help="How many GPUs to use for MetricX. Use -1 for all visible GPUs.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="MetricX: number of worker processes. 0=auto (== #devices). Can be > #devices (repeats devices, useful for testing on 1 GPU).",
    )

    parser.add_argument("--shard_rows", type=int, default=2_000_000, help="How many rows per Parquet shard.")
    parser.add_argument("--row_group_size", type=int, default=65536)
    parser.add_argument("--queue_max_batches", type=int, default=32, help="Max queued batches (input + output). Keep modest to limit RAM.")

    parser.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "NONE"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--datasets_json", type=str, default="", help="Optional: path to a JSON list of datasets.")
    parser.add_argument("--limit_dataset", type=str, default="", help="Optional: only use a single dataset.")
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Smoke test: score only the first N samples from the first dataset and print a summary.",
    )
    parser.add_argument("--test_samples", type=int, default=1024)
    parser.add_argument("--test_out_prefix", type=str, default="TEST_")

    args = parser.parse_args()

    datasets_to_use = DATASETS_TO_USE
    if args.datasets_json:
        datasets_to_use = json.loads(Path(args.datasets_json).read_text(encoding="utf-8"))
    if args.limit_dataset:
        datasets_to_use = [args.limit_dataset]
    if args.test_run:
        datasets_to_use = [datasets_to_use[0]]

    if not args.out_dir.strip():
        args.out_dir = f"outputs_{safe_name(args.model_name)}_parquet"

    if args.test_run:
        args.out_dir = f"{args.test_out_prefix}{args.out_dir}"
        if len(args.splits) > 1:
            args.splits = [args.splits[0]]

    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    stop_event = threading.Event()

    def _handle_sig(sig, frame):
        logger.warning("Received signal %s — stopping after current in-flight batches and flushing contiguous shards.", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    compression = choose_parquet_compression(args.compression)

    scorer_kind = args.scorer
    if scorer_kind == "auto":
        if "metricx" in args.model_name.lower():
            scorer_kind = "metricx_qe"
        else:
            scorer_kind = "mexma_cosine"

    # Decide devices/workers for MetricX
    devices = _parse_devices(args) if scorer_kind == "metricx_qe" else [args.device]
    if scorer_kind == "metricx_qe":
        # If user forces num_workers, expand devices (repeat round-robin)
        effective_workers = args.num_workers if args.num_workers and args.num_workers > 0 else len(devices)
        devices = _expand_devices_for_workers(devices, effective_workers)

    # Multi-worker MetricX enabled if >1 worker on CUDA (even if same GPU repeated)
    use_multi_worker_metricx = (
        scorer_kind == "metricx_qe"
        and len(devices) > 1
        and all(d.startswith("cuda") for d in devices)
    )

    scorer_single: Optional[BaseScorer] = None
    metric_names: List[str] = []

    if not use_multi_worker_metricx:
        device_single = args.device
        if scorer_kind == "mexma_cosine":
            scorer_single = MexmaCosineScorer(
                model_name=args.model_name,
                device=device_single,
                max_length=args.max_length,
                encode_batch_size=args.encode_batch_size,
                num_encode_workers=args.num_encode_workers,
                use_fp16=True,
            )
        else:
            scorer_single = MetricXQEScorer(
                model_name_or_path=args.model_name,
                device=device_single,
                batch_size=args.encode_batch_size,
                tokenizer_name=args.metricx_tokenizer,
                force_qe=True,
                max_input_length=args.metricx_max_input_length,
                fp16=True if args.metricx_fp16 or (device_single.startswith("cuda") and not args.metricx_bf16) else False,
                bf16=True if args.metricx_bf16 else False,
            )

        metric_names = scorer_single.metric_names
        logger.info(
            "Mode=SINGLE | Scorer=%s | metrics=%s | device=%s | compression=%s | out_dir=%s",
            scorer_kind, metric_names, device_single, compression, str(out_root)
        )
    else:
        metric_names = ["metricx_pred"]
        logger.info(
            "Mode=MULTI-WORKER MetricX | workers=%d | devices=%s | metric=%s | compression=%s | out_dir=%s",
            len(devices), devices, metric_names, compression, str(out_root)
        )
        # NOTE: Multi-worker on one GPU will likely be slower and use more VRAM. This is meant for testing.

    for dataset_name in datasets_to_use:
        if stop_event.is_set():
            break

        logger.info("=== DATASET: %s | subset=%s | splits=%s ===", dataset_name, args.subset, args.splits)

        for split in args.splits:
            if stop_event.is_set():
                break

            out_split_dir = out_root / safe_name(dataset_name) / safe_name(args.subset) / safe_name(split)
            ensure_dir(out_split_dir)

            try:
                ds = load_dataset(dataset_name, args.subset, split=split)
            except Exception as e:
                logger.warning("Failed to load %s / %s / %s: %s", dataset_name, args.subset, split, e)
                continue

            max_samples = DATASET_MAX_SAMPLES.get(dataset_name, None)

            resume_idx, resume_shard_id = find_resume_state(out_split_dir)

            if args.test_run:
                resume_idx, resume_shard_id = 0, 0
                max_samples = min(args.test_samples, max_samples) if max_samples is not None else args.test_samples

            total = len(ds)
            end_total = min(total, resume_idx + max_samples) if max_samples is not None else total
            if resume_idx >= end_total:
                logger.info("Split %s already processed (resume_idx=%d, end_total=%d). Skipping.", split, resume_idx, end_total)
                continue

            logger.info(
                "Resume: idx=%d -> end_total=%d (len=%d, limit=%s), next_shard_id=%d",
                resume_idx, end_total, total, str(max_samples), resume_shard_id
            )

            # Writer thread + queue (in main process)
            writer_q: "queue.Queue[Optional[BatchResult]]" = queue.Queue(maxsize=args.queue_max_batches)
            writer = ParquetShardWriter(
                out_split_dir=out_split_dir,
                q=writer_q,
                stop_event=stop_event,
                shard_rows=args.shard_rows,
                compression=compression,
                metric_names=metric_names,
                row_group_size=args.row_group_size,
            )
            writer.next_idx = resume_idx
            writer.next_shard_id = resume_shard_id

            writer_thread = threading.Thread(target=writer.run, daemon=True)
            writer_thread.start()

            pbar = tqdm(
                total=(end_total - resume_idx),
                desc=f"{dataset_name} [{split}]",
                unit="pairs",
                dynamic_ncols=True,
                smoothing=0.05,
            )

            t0 = time.perf_counter()
            processed = 0

            primary_metric = metric_names[0] if metric_names else None
            test_rows: List[Dict[str, Any]] = []

            if not use_multi_worker_metricx:
                # -------------------- SINGLE-PROCESS LOOP --------------------
                try:
                    assert scorer_single is not None

                    for i, j, batch in iter_dataset_slices(ds, resume_idx, args.batch_size, max_samples):
                        if stop_event.is_set():
                            break

                        en = batch.get(args.src_col)
                        pl = batch.get(args.tgt_col)
                        if en is None or pl is None:
                            raise KeyError(
                                f"Missing column '{args.src_col}' and/or '{args.tgt_col}'. Available: {list(batch.keys())}"
                            )

                        en = ["" if x is None else str(x) for x in en]
                        pl = ["" if x is None else str(x) for x in pl]

                        metrics = scorer_single.score_batch(en, pl)
                        for k, v in metrics.items():
                            metrics[k] = np.asarray(v, dtype=np.float32)

                        if args.test_run and primary_metric is not None:
                            scores = metrics[primary_metric]
                            for _en, _pl, _s in zip(en, pl, scores):
                                test_rows.append({"english": _en, "polish": _pl, "score": float(_s)})

                        idx = np.arange(i, j, dtype=np.int64)

                        writer_q.put(BatchResult(
                            start_idx=i,
                            end_idx=j,
                            idx=idx,
                            english=en,
                            polish=pl,
                            metrics=metrics,
                        ))

                        step = (j - i)
                        processed += step
                        pbar.update(step)

                        dt = time.perf_counter() - t0
                        if dt > 0:
                            pbar.set_postfix({"pairs/s": f"{processed / dt:,.0f}"})

                except KeyboardInterrupt:
                    stop_event.set()
                finally:
                    pbar.close()
                    try:
                        writer_q.put(None)
                    except Exception:
                        pass
                    writer_q.join()
                    writer_thread.join(timeout=300)

            else:
                # -------------------- MULTI-WORKER MetricX LOOP --------------------
                mp_ctx = mp.get_context("spawn")
                in_q: "mp.Queue" = mp_ctx.Queue(maxsize=args.queue_max_batches)
                out_q: "mp.Queue" = mp_ctx.Queue(maxsize=args.queue_max_batches)
                stop_ev_mp: "mp.Event" = mp_ctx.Event()

                # Propagate main stop_event to mp stop event
                def _stop_watcher():
                    while not stop_event.is_set():
                        time.sleep(0.1)
                    stop_ev_mp.set()

                watcher_thread = threading.Thread(target=_stop_watcher, daemon=True)
                watcher_thread.start()

                workers: List[mp.Process] = []
                fp16 = True if args.metricx_fp16 or (not args.metricx_bf16) else False
                bf16 = True if args.metricx_bf16 else False

                for rank, dev in enumerate(devices):
                    p = mp_ctx.Process(
                        target=_metricx_worker,
                        args=(
                            rank,
                            dev,
                            args.model_name,
                            args.metricx_tokenizer,
                            args.encode_batch_size,
                            args.metricx_max_input_length,
                            fp16,
                            bf16,
                            in_q,
                            out_q,
                            stop_ev_mp,
                        ),
                        daemon=True,
                    )
                    p.start()
                    workers.append(p)

                pending: Dict[int, BatchResult] = {}
                next_expected = resume_idx
                done_workers = 0
                had_error = False
                error_info = None

                def _drain_contiguous():
                    nonlocal next_expected, processed
                    while next_expected in pending:
                        br = pending.pop(next_expected)
                        writer_q.put(br)

                        step = br.end_idx - br.start_idx
                        processed += step
                        pbar.update(step)

                        if args.test_run and primary_metric is not None:
                            scores = br.metrics[primary_metric]
                            for _en, _pl, _s in zip(br.english, br.polish, scores):
                                test_rows.append({"english": _en, "polish": _pl, "score": float(_s)})

                        next_expected = br.end_idx

                        dt = time.perf_counter() - t0
                        if dt > 0:
                            pbar.set_postfix({"pairs/s": f"{processed / dt:,.0f}"})

                enqueue_done = threading.Event()

                def _enqueue_loop():
                    try:
                        for i, j, batch in iter_dataset_slices(ds, resume_idx, args.batch_size, max_samples):
                            if stop_event.is_set():
                                break

                            en = batch.get(args.src_col)
                            pl = batch.get(args.tgt_col)
                            if en is None or pl is None:
                                raise KeyError(
                                    f"Missing column '{args.src_col}' and/or '{args.tgt_col}'. Available: {list(batch.keys())}"
                                )

                            en = ["" if x is None else str(x) for x in en]
                            pl = ["" if x is None else str(x) for x in pl]

                            task = {"start_idx": i, "end_idx": j, "english": en, "polish": pl}
                            in_q.put(task)

                        # Stop sentinels
                        for _ in workers:
                            in_q.put(None)

                    except Exception:
                        stop_event.set()
                        stop_ev_mp.set()
                        logger.error("Enqueue loop failed:\n%s", traceback.format_exc())
                        try:
                            for _ in workers:
                                in_q.put(None)
                        except Exception:
                            pass
                    finally:
                        enqueue_done.set()

                enqueue_thread = threading.Thread(target=_enqueue_loop, daemon=True)
                enqueue_thread.start()

                try:
                    while True:
                        if had_error:
                            break

                        if done_workers == len(workers) and enqueue_done.is_set():
                            _drain_contiguous()
                            break

                        msg_type, payload = out_q.get()

                        if msg_type == "result":
                            br: BatchResult = payload
                            pending[br.start_idx] = br
                            _drain_contiguous()

                        elif msg_type == "done":
                            done_workers += 1

                        elif msg_type == "error":
                            had_error = True
                            error_info = payload
                            stop_event.set()
                            stop_ev_mp.set()
                            break

                except KeyboardInterrupt:
                    stop_event.set()
                    stop_ev_mp.set()
                finally:
                    pbar.close()

                    if had_error and error_info is not None:
                        rank, msg, tb = error_info
                        logger.error("Worker %s failed: %s\n%s", rank, msg, tb)

                    # DO NOT write non-contiguous pending results (would break resume correctness)
                    try:
                        writer_q.put(None)
                    except Exception:
                        pass
                    writer_q.join()
                    writer_thread.join(timeout=300)

                    for p in workers:
                        try:
                            p.join(timeout=30)
                        except Exception:
                            pass
                    for p in workers:
                        if p.is_alive():
                            try:
                                p.terminate()
                            except Exception:
                                pass

                    try:
                        enqueue_thread.join(timeout=60)
                    except Exception:
                        pass

            dt = time.perf_counter() - t0
            logger.info(
                "DONE [%s | %s] processed=%d in %.1fs (%.1f pairs/s). Out: %s",
                dataset_name, split, processed, dt, processed / max(dt, 1e-9), str(out_split_dir)
            )

            if args.test_run and primary_metric is not None and test_rows:
                test_rows_sorted = sorted(test_rows, key=lambda r: r["score"])
                lowest5 = test_rows_sorted[:5]
                highest5 = test_rows_sorted[-5:][::-1]

                logger.info("=== TEST RUN SUMMARY ===")
                logger.info(
                    "Dataset=%s | split=%s | samples=%d | metric=%s",
                    dataset_name, split, len(test_rows), primary_metric
                )

                print("\n--- TOP 5 (highest score=worst translation) ---")
                for k, r in enumerate(highest5, 1):
                    print(
                        f"{k:02d}. score={r['score']:.6f}\n"
                        f"    EN: {r['english']}\n"
                        f"    PL: {r['polish']}\n"
                    )

                print("\n--- BOTTOM 5 (lowest score=best translation) ---")
                for k, r in enumerate(lowest5, 1):
                    print(
                        f"{k:02d}. score={r['score']:.6f}\n"
                        f"    EN: {r['english']}\n"
                        f"    PL: {r['polish']}\n"
                    )
                return

    logger.info("Done.")


if __name__ == "__main__":
    main()
