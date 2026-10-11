"""Explicit PyTorch training loop and bounded, single-job orchestration."""
import copy
import json
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import (MAX_CHECKPOINTS, MAX_PARAMETERS, MAX_RUNTIME_SECONDS,
                     MAX_STORAGE_BYTES, MAX_TASKS, GenerateRequest, PredictRequest, TrainConfig, TrainRequest)
from .data import append_token, prepare_data, read_corpus, tokenize
from .model import TinyTransformer, parameter_count


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail


@dataclass
class Job:
    id: str
    config: TrainConfig
    text: str
    vocabulary: list[str]
    data: dict
    device: str = "cpu"
    status: str = "queued"
    step: int = 0
    target_steps: int = 0
    elapsed_seconds: float = 0.0
    metrics: list = field(default_factory=list)
    attention: dict | None = None
    checkpoint_id: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    model: TinyTransformer | None = None
    optimizer: torch.optim.Optimizer | None = None
    rng_state: torch.Tensor | None = None
    cuda_rng_state: torch.Tensor | None = None
    peak_gpu_memory_mb: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)


@torch.inference_mode()
def evaluate(model, loader, *, max_batches=None, should_stop=lambda: False):
    model.eval()
    device = next(model.parameters()).device
    total_loss, correct, count = 0.0, 0, 0
    for index, (x, y) in enumerate(loader):
        if (max_batches is not None and index >= max_batches) or should_stop():
            break
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total_loss += F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum").item()
        correct += int((logits.argmax(-1) == y).sum())
        count += y.numel()
    if not count:
        return None
    loss = total_loss / count
    if not math.isfinite(loss) or loss > 80:
        raise ValueError("评估 loss 数值不稳定，请降低学习率")
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": correct / count,
            "tokens": count}


@torch.inference_mode()
def attention_view(model, tokens, vocabulary, layer=0, head=0):
    model.eval()
    device = next(model.parameters()).device
    logits, weights = model(tokens.unsqueeze(0).to(device), return_attention=True)
    return logits, {"tokens": [vocabulary[int(i)] for i in tokens], "layer": layer, "head": head,
                    "weights": weights[layer][0, head].tolist()}


class TrainingManager:
    def __init__(self, storage: Path):
        self.storage = storage
        storage.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch-training")
        self.closed = False
        # Local CPU baseline deliberately reserves most of the machine for the user.
        torch.set_num_threads(2)

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                job.cancel.set()
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.jobs.clear()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

    @staticmethod
    def hardware():
        available = torch.cuda.is_available()
        return {"cuda_available": available, "devices": ["cpu", "cuda"] if available else ["cpu"],
                "gpu_name": torch.cuda.get_device_name(0) if available else None,
                "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2 if available else 0,
                "gpu_allocator_limit_mb": 2048, "pytorch_version": str(torch.__version__)}

    @staticmethod
    def _device(choice):
        available = torch.cuda.is_available()
        if choice == "cuda" and not available:
            raise ServiceError(422, "CUDA 不可用，请安装 CUDA 版 PyTorch 并检查 NVIDIA 驱动，或选择 CPU")
        return "cuda" if choice != "cpu" and available else "cpu"

    @staticmethod
    def _gpu_budget():
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.25, 2 * 1024**3 / total), 0)
        torch.cuda.reset_peak_memory_stats(0)

    def _idle(self):
        if self.closed:
            raise ServiceError(503, "训练服务正在关闭")
        if any(j.status in {"queued", "running", "stopping"} for j in self.jobs.values()):
            raise ServiceError(409, "已有训练任务运行中；请停止或等待完成后再操作模型")

    def _room(self):
        while len(self.jobs) >= MAX_TASKS:
            self.jobs.pop(next(iter(self.jobs)))

    def _job(self, task_id):
        if task_id not in self.jobs:
            raise ServiceError(404, "训练任务不存在或已从内存释放，请加载 checkpoint")
        return self.jobs[task_id]

    def snapshot(self, task_id):
        with self.lock:
            job = self._job(task_id)
            return copy.deepcopy({"id": job.id, "status": job.status, "config": job.config.model_dump(),
                "step": job.step, "target_steps": job.target_steps, "elapsed_seconds": job.elapsed_seconds,
                "parameter_count": parameter_count(job.config, len(job.vocabulary)), "data": job.data,
                "device": job.device, "peak_gpu_memory_mb": job.peak_gpu_memory_mb,
                "metrics": job.metrics, "attention": job.attention, "checkpoint_id": job.checkpoint_id,
                "error": job.error, "stop_reason": job.stop_reason})

    def create(self, request: TrainRequest):
        # Check the global gate before preparing potentially large tensors.
        with self.lock:
            self._idle()
            device = self._device(request.config.device)
            text = request.text if request.text is not None else read_corpus(request.config.dataset)
            train, _, vocabulary, data = prepare_data(text, request.config.sequence_length,
                                                       request.config.validation_ratio, request.config.tokenizer)
            if parameter_count(request.config, len(vocabulary)) > MAX_PARAMETERS:
                raise ServiceError(422, "模型超过 300 万参数预算")
            self._room()
            batches = math.ceil(len(train) / request.config.batch_size)
            target = min(request.config.max_steps, (request.config.epochs or 2000) * batches)
            job = Job(uuid.uuid4().hex, request.config.model_copy(), text, vocabulary, data,
                      target_steps=target, device=device)
            self.jobs[job.id] = job
            self.executor.submit(self._train, job)
            return self.snapshot(job.id)

    def stop(self, task_id):
        with self.lock:
            job = self._job(task_id)
            if job.status in {"queued", "running", "stopping"}:
                job.cancel.set()
                job.status = "stopping"
            return self.snapshot(task_id)

    def resume(self, task_id, max_steps):
        with self.lock:
            self._idle()
            job = self._job(task_id)
            if job.status not in {"completed", "stopped", "loaded"} or job.model is None:
                raise ServiceError(409, "只有已完成、停止或加载的模型可以继续训练")
            if max_steps <= job.step:
                raise ServiceError(422, "max_steps 是累计目标步数，必须大于当前 step")
            job.target_steps = max_steps
            job.config = job.config.model_copy(update={"max_steps": max_steps, "epochs": None})
            job.cancel.clear()
            job.status, job.error, job.stop_reason = "queued", None, None
            self.executor.submit(self._train, job)
            return self.snapshot(task_id)

    def _train(self, job):
        started, previous_elapsed = time.perf_counter(), job.elapsed_seconds
        try:
            with self.lock:
                job.status = "running"
            train, validation, _, _ = prepare_data(job.text, job.config.sequence_length,
                                                   job.config.validation_ratio, job.config.tokenizer)
            if job.device == "cuda":
                self._gpu_budget()
            if job.model is None:
                torch.manual_seed(job.config.seed)
                job.model = TinyTransformer(job.config, len(job.vocabulary)).to(job.device)
                job.optimizer = torch.optim.AdamW(job.model.parameters(), lr=job.config.learning_rate)
            elif job.rng_state is not None:
                torch.set_rng_state(job.rng_state)
                if job.device == "cuda" and job.cuda_rng_state is not None:
                    torch.cuda.set_rng_state(job.cuda_rng_state)
            validation_loader = DataLoader(validation, batch_size=job.config.batch_size, shuffle=False)
            train_eval_loader = DataLoader(train, batch_size=job.config.batch_size, shuffle=False)
            batches = math.ceil(len(train) / job.config.batch_size)

            def should_stop():
                if time.perf_counter() - started >= MAX_RUNTIME_SECONDS:
                    job.stop_reason = "180 秒单次运行预算已用完，可从 checkpoint 继续训练"
                    job.cancel.set()
                return job.cancel.is_set()

            def report(optimization_loss=None, *, final=False):
                # At cancellation, finish a bounded evaluation so the last checkpoint has valid metrics.
                stop = (lambda: False) if final else should_stop
                train_stats = evaluate(job.model, train_eval_loader, max_batches=8, should_stop=stop)
                val_stats = evaluate(job.model, validation_loader, should_stop=stop)
                if train_stats is None or val_stats is None or (should_stop() and not final):
                    return
                if not all(math.isfinite(s["loss"]) for s in (train_stats, val_stats)):
                    raise ValueError("loss 非有限值，请降低学习率")
                _, attention = attention_view(job.model, validation[0][0][:32], job.vocabulary)
                point = {"step": job.step, "epoch": job.step / batches,
                    "train_loss": train_stats["loss"], "validation_loss": val_stats["loss"],
                    "perplexity": val_stats["perplexity"], "accuracy": val_stats["accuracy"],
                    "learning_rate": job.optimizer.param_groups[0]["lr"],
                    "optimization_loss": optimization_loss,
                    "train_eval_tokens": train_stats["tokens"], "validation_tokens": val_stats["tokens"],
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - started}
                with self.lock:
                    if job.metrics and job.metrics[-1]["step"] == job.step:
                        job.metrics[-1] = point
                    else:
                        job.metrics.append(point)
                    job.attention = attention
                    job.elapsed_seconds = point["elapsed_seconds"]
                    if job.device == "cuda":
                        job.peak_gpu_memory_mb = torch.cuda.max_memory_allocated(0) / 1024**2

            if not job.metrics:
                report()
            losses = []
            while job.step < job.target_steps and not should_stop():
                epoch, offset = divmod(job.step, batches)
                # Epoch-local deterministic shuffling + cursor enable exact optimizer continuation.
                generator = torch.Generator().manual_seed(job.config.seed + epoch)
                loader = DataLoader(train, batch_size=job.config.batch_size, shuffle=True,
                                    generator=generator, num_workers=0)
                for batch_index, (x, y) in enumerate(loader):
                    if batch_index < offset:
                        continue
                    if should_stop() or job.step >= job.target_steps:
                        break
                    job.model.train()
                    x, y = x.to(job.device), y.to(job.device)
                    job.optimizer.zero_grad(set_to_none=True)
                    logits, _ = job.model(x)
                    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
                    if not torch.isfinite(loss):
                        raise ValueError("loss 非有限值，请降低学习率")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(job.model.parameters(), 1.0, error_if_nonfinite=True)
                    job.optimizer.step()
                    losses.append(loss.item())
                    with self.lock:
                        job.step += 1
                        job.elapsed_seconds = previous_elapsed + time.perf_counter() - started
                    if job.step % job.config.eval_interval == 0:
                        report(sum(losses) / len(losses))
                        losses.clear()
            report(sum(losses) / len(losses) if losses else None, final=True)
            job.rng_state = torch.get_rng_state()
            if job.device == "cuda":
                job.cuda_rng_state = torch.cuda.get_rng_state()
                torch.cuda.synchronize()
            with self.lock:
                job.elapsed_seconds = previous_elapsed + time.perf_counter() - started
                self._save(job)
                job.status = "stopped" if job.cancel.is_set() else "completed"
                if job.status == "stopped" and job.stop_reason is None:
                    job.stop_reason = "用户停止训练"
        except Exception as exc:
            with self.lock:
                job.error = "GPU 显存不足或超过预算，请减小 batch / 上下文 / 模型，或选择 CPU" if isinstance(exc, torch.cuda.OutOfMemoryError) else str(exc)
                job.status = "failed"
                job.elapsed_seconds = previous_elapsed + time.perf_counter() - started

    def checkpoints(self):
        with self.lock:
            results = []
            for path in self.storage.glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    TrainConfig.model_validate(item["config"])
                    if (not isinstance(item["step"], int) or not isinstance(item["size_bytes"], int)
                            or not isinstance(item["created_at"], (int, float))
                            or not math.isfinite(item["created_at"])):
                        continue
                    if self._checkpoint_path(item["id"]).is_file():
                        results.append(item)
                except (ValueError, KeyError, TypeError, ServiceError):
                    continue
            return sorted(results, key=lambda item: item["created_at"], reverse=True)

    def _checkpoint_path(self, checkpoint_id):
        if not re.fullmatch(r"[0-9a-f]{32}", checkpoint_id):
            raise ServiceError(404, "checkpoint 不存在")
        return self.storage / f"{checkpoint_id}.pt"

    def _save(self, job):
        if job.model is None:
            raise ServiceError(409, "模型尚未初始化")
        existing_bytes = sum(p.stat().st_size for p in self.storage.iterdir() if p.is_file())
        if len(self.checkpoints()) >= MAX_CHECKPOINTS or existing_bytes >= MAX_STORAGE_BYTES:
            raise ServiceError(409, "checkpoint 存储预算已满，请在服务停止后清理存档目录")
        checkpoint_id = uuid.uuid4().hex
        path = self._checkpoint_path(checkpoint_id)
        temporary = path.with_suffix(".tmp")
        payload = {"format_version": 2, "config": job.config.model_dump(), "text": job.text,
            "vocabulary": job.vocabulary, "data": job.data, "model": job.model.state_dict(),
            "optimizer": job.optimizer.state_dict(), "rng_state": job.rng_state,
            "cuda_rng_state": job.cuda_rng_state, "training_device": job.device,
            "peak_gpu_memory_mb": job.peak_gpu_memory_mb,
            "step": job.step, "metrics": job.metrics, "elapsed_seconds": job.elapsed_seconds,
            "pytorch_version": str(torch.__version__)}
        try:
            torch.save(payload, temporary)
            if existing_bytes + temporary.stat().st_size > MAX_STORAGE_BYTES:
                raise ServiceError(409, "checkpoint 超过 200 MB 存储预算")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        info = {"id": checkpoint_id, "task_id": job.id, "step": job.step,
                "created_at": time.time(), "config": job.config.model_dump(),
                "metrics": job.metrics[-1] if job.metrics else None,
                "size_bytes": path.stat().st_size, "data_sha256": job.data["sha256"]}
        info["device"] = job.device
        manifest = path.with_suffix(".json")
        manifest_tmp = manifest.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(manifest_tmp, manifest)
        job.checkpoint_id = checkpoint_id
        return info

    def save(self, task_id):
        with self.lock:
            self._idle()
            return self._save(self._job(task_id))

    def load(self, checkpoint_id, device_choice="auto"):
        with self.lock:
            self._idle()
            device = self._device(device_choice)
            if device == "cuda":
                self._gpu_budget()
            path = self._checkpoint_path(checkpoint_id)
            if not path.is_file():
                raise ServiceError(404, "checkpoint 不存在")
            if path.stat().st_size > MAX_STORAGE_BYTES:
                raise ServiceError(422, "checkpoint 超过大小限制")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["format_version"] not in {1, 2}:
                    raise ValueError("不支持的 checkpoint 版本")
                config = TrainConfig.model_validate(payload["config"])
                config = config.model_copy(update={"device": device_choice})
                _, validation, vocabulary, data = prepare_data(payload["text"], config.sequence_length,
                                                               config.validation_ratio, config.tokenizer)
                if vocabulary != payload["vocabulary"] or any(data.get(k) != v for k, v in payload["data"].items()):
                    raise ValueError("checkpoint 词表或数据摘要不一致")
                if parameter_count(config, len(vocabulary)) > MAX_PARAMETERS:
                    raise ValueError("checkpoint 参数预算超限")
                model = TinyTransformer(config, len(vocabulary)).to(device)
                model.load_state_dict(payload["model"])
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise ValueError("checkpoint 权重包含非有限值")
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
                optimizer.load_state_dict(payload["optimizer"])
                job = Job(uuid.uuid4().hex, config, payload["text"], vocabulary, data,
                    status="loaded", step=payload["step"], target_steps=config.max_steps,
                    elapsed_seconds=payload["elapsed_seconds"], metrics=payload["metrics"],
                    checkpoint_id=checkpoint_id, model=model, optimizer=optimizer,
                    rng_state=payload["rng_state"], device=device,
                    cuda_rng_state=payload.get("cuda_rng_state"),
                    peak_gpu_memory_mb=payload.get("peak_gpu_memory_mb", 0.0))
                _, job.attention = attention_view(model, validation[0][0][:32], vocabulary)
            except Exception as exc:
                raise ServiceError(422, f"无法加载 checkpoint：{exc}") from exc
            self._room()
            self.jobs[job.id] = job
            return self.snapshot(job.id)

    def checkpoint_file(self, checkpoint_id):
        path = self._checkpoint_path(checkpoint_id)
        if not path.is_file():
            raise ServiceError(404, "checkpoint 不存在")
        return path

    def _inference_input(self, task_id, request):
        self._idle()
        job = self._job(task_id)
        if job.model is None or job.status not in {"completed", "stopped", "loaded"}:
            raise ServiceError(409, "请先完成训练或加载 checkpoint")
        if request.layer >= job.config.layers or request.head >= job.config.heads:
            raise ServiceError(422, "所选 layer/head 超出模型结构")
        tokens = request.context_tokens or tokenize(request.prompt, job.config.tokenizer)
        if not tokens:
            raise ServiceError(422, "提示至少须包含一个有效词元")
        lookup = {token: index for index, token in enumerate(job.vocabulary)}
        unknown = sorted(set(tokens) - set(lookup))
        if unknown or "<unk>" in tokens:
            raise ServiceError(422, f"提示含训练词表之外的词元：{' / '.join(unknown[:10])}")
        return job, [lookup[token] for token in tokens][-job.config.sequence_length:]

    def _prediction(self, job, ids, request):
        job.model.eval()
        x = torch.tensor([ids[-job.config.sequence_length:]], dtype=torch.long, device=job.device)
        logits, weights = job.model(x, return_attention=True)
        raw_scores = logits[0, -1].clone()
        raw_scores[0] = float("-inf")
        model_probabilities = F.softmax(raw_scores, dim=-1)
        scores = raw_scores / request.temperature
        if request.top_k:
            cutoff = torch.topk(scores, min(request.top_k, len(job.vocabulary) - 1)).values[-1]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        values, raw_values = probabilities.tolist(), model_probabilities.tolist()
        candidates = sorted([{"token": token, "token_id": i, "probability": values[i],
                              "model_probability": raw_values[i]} for i, token in enumerate(job.vocabulary) if i],
                            key=lambda row: (-row["probability"], -row["model_probability"]))
        shown = min(32, x.shape[1])
        result = {"candidates": candidates, "context_tokens": [job.vocabulary[i] for i in ids],
                  "tokenizer": job.config.tokenizer, "temperature": request.temperature, "top_k": request.top_k,
                  "attention": {"tokens": [job.vocabulary[i] for i in ids[:shown]], "layer": request.layer,
                                "head": request.head, "weights": weights[request.layer][0, request.head, :shown, :shown].tolist()}}
        return result, probabilities

    @torch.inference_mode()
    def predict(self, task_id, request: PredictRequest):
        with self.lock:
            job, ids = self._inference_input(task_id, request)
            result, _ = self._prediction(job, ids, request)
            return result

    @torch.inference_mode()
    def step(self, task_id, request: PredictRequest):
        with self.lock:
            job, ids = self._inference_input(task_id, request)
            _, probabilities = self._prediction(job, ids, request)
            generator = torch.Generator(device=job.device).manual_seed(request.seed)
            next_id = int(torch.multinomial(probabilities, 1, generator=generator))
            after_ids = (ids + [next_id])[-job.config.sequence_length:]
            after, _ = self._prediction(job, after_ids, request)
            return {"next_token": job.vocabulary[next_id], "probability": float(probabilities[next_id]),
                    "seed": request.seed, "prediction": after}

    def generate(self, task_id, request: GenerateRequest):
        with self.lock, torch.inference_mode():
            job, ids = self._inference_input(task_id, request)
            context = torch.tensor(ids[-min(job.config.sequence_length, 32):], dtype=torch.long)
            _, attention = attention_view(job.model, context, job.vocabulary, request.layer, request.head)
            generator = torch.Generator(device=job.device).manual_seed(request.seed)
            new_ids = []
            job.model.eval()
            for _ in range(request.max_new_tokens):
                x = torch.tensor([ids[-job.config.sequence_length:]], dtype=torch.long, device=job.device)
                logits, _ = job.model(x)
                scores = logits[0, -1] / request.temperature
                scores[0] = float("-inf")  # <unk> is an evaluation fallback, never generated.
                if request.top_k:
                    cutoff = torch.topk(scores, min(request.top_k, len(job.vocabulary) - 1)).values[-1]
                    scores = scores.masked_fill(scores < cutoff, float("-inf"))
                next_id = int(torch.multinomial(F.softmax(scores, dim=-1), 1, generator=generator))
                ids.append(next_id)
                new_ids.append(next_id)
                if request.stop_after_sentence and job.vocabulary[next_id] in {".", "!", "?", "。", "！", "？"}:
                    break
            output = request.prompt
            for i in new_ids:
                output = append_token(output, job.vocabulary[i], job.config.tokenizer)
            generated = output[len(request.prompt):]
            return {"task_id": task_id, "prompt": request.prompt, "generated_text": generated,
                    "text": request.prompt + generated, "attention": attention,
                    "context_window": job.config.sequence_length, "seed": request.seed}
