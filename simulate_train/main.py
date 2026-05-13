from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table


RAINBOW_COLORS = (
    "\033[38;5;196m",
    "\033[38;5;208m",
    "\033[38;5;226m",
    "\033[38;5;46m",
    "\033[38;5;51m",
    "\033[38;5;39m",
    "\033[38;5;129m",
)
RESET_COLOR = "\033[0m"
LOG_STYLES = ("trainer", "deepspeed", "vllm", "stable-diffusion")
SCENARIOS = ("normal", "llm-pretrain", "finetune", "diffusion")
CHAOS_LEVELS = (0, 1, 2)
STAGE_NAMES = ("bootstrap", "warmup", "train", "incident", "recovery", "eval", "checkpoint", "summary")


@dataclass(frozen=True)
class IncidentSpec:
    level: str
    message: str
    at_pct: float


@dataclass(frozen=True)
class StageSpec:
    name: str
    start_pct: float
    end_pct: float
    scenario: str = "normal"
    chaos_level: int = 1
    events: tuple[IncidentSpec, ...] = ()


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str
    kind: str = "professional"
    log_style: str = "trainer"
    seed: Optional[int] = 42
    steps: int = 240
    step_delay: float = 0.03
    speed_jitter: float = 0.18
    log_every: int = 12
    save_every: int = 80
    save_delay: float = 0.15
    project_name: str = "trainfake"
    run_name: str = "cinematic-run"
    loss_start: float = 1.5
    loss_min: float = 0.1
    acc_start: float = 0.5
    oscillation: float = 0.05
    stages: tuple[StageSpec, ...] = field(default_factory=tuple)


@dataclass
class RunSummary:
    spec_name: str
    steps: int
    final_metrics: Optional["TrainingMetrics"] = None
    best_checkpoint: str = "none"
    incidents: int = 0
    recoveries: int = 0
    elapsed_seconds: float = 0.0


def incident(level: str, message: str, at_pct: float) -> IncidentSpec:
    return IncidentSpec(level=level, message=message, at_pct=at_pct)


def default_stages(scenario: str = "normal", chaos_level: int = 1) -> tuple[StageSpec, ...]:
    return (
        StageSpec("bootstrap", 0.0, 0.06, scenario, 0),
        StageSpec("warmup", 0.06, 0.22, scenario, max(0, chaos_level - 1)),
        StageSpec("train", 0.22, 0.72, scenario, chaos_level),
        StageSpec("incident", 0.72, 0.82, scenario, min(2, chaos_level + 1)),
        StageSpec("recovery", 0.82, 0.92, scenario, chaos_level),
        StageSpec("eval", 0.92, 0.98, scenario, max(0, chaos_level - 1)),
        StageSpec("summary", 0.98, 1.0, scenario, 0),
    )


PRESETS: dict[str, RunSpec] = {
    "llama-70b-pretrain": RunSpec(
        name="llama-70b-pretrain",
        description="Multi-node LLM pretraining run with warmup, scaler drama, and sharded checkpoints.",
        kind="professional",
        log_style="deepspeed",
        seed=70,
        steps=260,
        step_delay=0.025,
        log_every=13,
        save_every=65,
        project_name="frontier-lab",
        run_name="llama-70b-pretrain-rank0",
        loss_start=2.4,
        loss_min=0.18,
        acc_start=0.38,
        stages=(
            StageSpec("bootstrap", 0.0, 0.06, "llm-pretrain", 0),
            StageSpec("warmup", 0.06, 0.2, "llm-pretrain", 1),
            StageSpec("train", 0.2, 0.68, "llm-pretrain", 1),
            StageSpec(
                "incident",
                0.68,
                0.8,
                "llm-pretrain",
                2,
                (
                    incident("WARNING", "NCCL watchdog noticed a slow allreduce; rank0 keeps the job alive", 0.71),
                    incident("ERROR", "fp16 overflow on microbatch 3; optimizer step skipped and scaler restored", 0.76),
                ),
            ),
            StageSpec("recovery", 0.8, 0.91, "llm-pretrain", 1),
            StageSpec("eval", 0.91, 0.98, "llm-pretrain", 0),
            StageSpec("summary", 0.98, 1.0, "llm-pretrain", 0),
        ),
    ),
    "sdxl-lora": RunSpec(
        name="sdxl-lora",
        description="Diffusion fine-tuning run with EMA updates, preview sampling, and latent cache churn.",
        kind="professional",
        log_style="stable-diffusion",
        seed=1024,
        steps=220,
        step_delay=0.025,
        log_every=11,
        save_every=55,
        project_name="diffusion-lab",
        run_name="sdxl-lora-nightly",
        loss_start=0.92,
        loss_min=0.04,
        acc_start=0.45,
        stages=default_stages("diffusion", 1),
    ),
    "vllm-launch": RunSpec(
        name="vllm-launch",
        description="Serving-style launch rehearsal with request queues, KV cache pressure, and throughput recovery.",
        kind="professional",
        log_style="vllm",
        seed=8080,
        steps=180,
        step_delay=0.02,
        log_every=9,
        save_every=0,
        project_name="inference",
        run_name="vllm-launch-window",
        loss_start=1.1,
        loss_min=0.08,
        acc_start=0.52,
        stages=default_stages("llm-pretrain", 1),
    ),
    "bert-finetune": RunSpec(
        name="bert-finetune",
        description="Classic supervised fine-tuning run with eval jumps and mild overfit warnings.",
        kind="professional",
        log_style="trainer",
        seed=12,
        steps=160,
        step_delay=0.02,
        log_every=8,
        save_every=40,
        project_name="nlp-baselines",
        run_name="bert-finetune-imdb",
        loss_start=1.25,
        loss_min=0.06,
        acc_start=0.58,
        stages=default_stages("finetune", 1),
    ),
    "boss-is-watching": RunSpec(
        name="boss-is-watching",
        description="A believable training run tuned for maximum terminal seriousness in minimal time.",
        kind="parody",
        log_style="deepspeed",
        seed=404,
        steps=120,
        step_delay=0.018,
        log_every=6,
        save_every=30,
        project_name="urgent-demo",
        run_name="quarterly-review-safe-run",
        loss_start=1.8,
        loss_min=0.12,
        acc_start=0.49,
        stages=default_stages("llm-pretrain", 2),
    ),
    "deadline-finetune": RunSpec(
        name="deadline-finetune",
        description="A compact adapter fine-tune that looks calm even when the deadline does not.",
        kind="parody",
        log_style="trainer",
        seed=515,
        steps=140,
        step_delay=0.018,
        log_every=7,
        save_every=35,
        project_name="last-minute",
        run_name="adapter-before-standup",
        loss_start=1.4,
        loss_min=0.07,
        acc_start=0.54,
        stages=default_stages("finetune", 2),
    ),
}


@dataclass(frozen=True)
class TrainingMetrics:
    loss: float
    val_loss: float
    lm_loss: float
    aux_loss: float
    perplexity: float
    accuracy: float
    auc: float
    f1: float
    precision: float
    recall: float
    learning_rate: float
    grad_norm: float
    gpu_memory_gb: float
    gpu_utilization: float
    loss_scale: float
    samples_per_second: float
    tokens_per_second: float
    requests_running: int
    requests_waiting: int
    gpu_cache_usage: float
    prefix_cache_hit_rate: float
    ttft_ms: float
    tpot_ms: float
    fwd_latency_ms: float
    bwd_latency_ms: float
    allreduce_latency_ms: float
    optimizer_step_ms: float
    ema_decay: float
    diffusion_timestep: int
    noise_offset: float
    snr_gamma: float


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trainfake",
        description="Run cinematic AI training logs for demos, tests, teaching, and suspiciously serious terminals.",
    )
    subparsers = parser.add_subparsers(dest="command")

    presets_parser = subparsers.add_parser("presets", help="List built-in cinematic run presets.")
    presets_parser.set_defaults(command="presets")

    run_parser = subparsers.add_parser("run", help="Run a built-in preset or a YAML run script.")
    run_parser.add_argument("preset", nargs="?", help="Built-in preset name.")
    run_parser.add_argument("--config", help="Path to a YAML run script.")
    run_parser.add_argument("--steps", type=int, help="Override total simulated steps.")
    run_parser.add_argument("--step-delay", type=float, help="Override seconds to wait between samples.")
    run_parser.add_argument("--seed", type=int, help="Override the run seed.")
    run_parser.add_argument("--log-every", type=int, help="Override metric log cadence.")
    run_parser.add_argument("--save-every", type=int, help="Override checkpoint cadence; use 0 to disable.")
    run_parser.add_argument("--rainbow", action="store_true", help="Enable ANSI coloring inside log payloads.")
    run_parser.set_defaults(command="run")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be greater than 0")
    if args.loss_start < 0:
        raise ValueError("--loss-start must be non-negative")
    if args.loss_min < 0:
        raise ValueError("--loss-min must be non-negative")
    if args.loss_start < args.loss_min:
        raise ValueError("--loss-start must be greater than or equal to --loss-min")
    if not 0 <= args.acc_start <= 0.99:
        raise ValueError("--acc-start must be between 0 and 0.99")
    if args.oscillation < 0:
        raise ValueError("--oscillation must be non-negative")
    if args.step_delay < 0:
        raise ValueError("--step-delay must be non-negative")
    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than 0")
    if args.scenario not in SCENARIOS:
        raise ValueError(f"--scenario must be one of: {', '.join(SCENARIOS)}")
    if args.chaos_level not in CHAOS_LEVELS:
        raise ValueError("--chaos-level must be 0, 1, or 2")
    if args.save_every < 0:
        raise ValueError("--save-every must be non-negative")
    if args.save_delay < 0:
        raise ValueError("--save-delay must be non-negative")
    if args.speed_jitter < 0:
        raise ValueError("--speed-jitter must be non-negative")
    if not args.project_name:
        raise ValueError("--project-name must not be empty")
    if not args.run_name:
        raise ValueError("--run-name must not be empty")


def simulated_metrics(
    step: int,
    steps: int,
    loss_start: float,
    loss_min: float,
    acc_start: float,
    oscillation: float,
    scenario: str = "normal",
    rng: Optional[random.Random] = None,
) -> TrainingMetrics:
    rng = rng or random.Random()
    progress = step / steps
    convergence_point = max(1, int(steps * 0.75))

    if step <= convergence_point:
        loss_progress = step / convergence_point
        loss = loss_start - (loss_start - loss_min) * loss_progress
    else:
        wave = math.sin((step - convergence_point) * math.pi / 40)
        loss = loss_min + oscillation * wave

    acc = acc_start + (0.99 - acc_start) * (1 - math.exp(-4 * progress))
    loss = max(loss_min, loss)
    accuracy = min(0.99, acc)
    validation_gap = 0.04 + 0.03 * (1 - progress) + 0.015 * math.sin(step * 0.017)
    val_loss = max(loss_min, loss + validation_gap)
    lm_loss = max(loss_min, loss * (0.86 + 0.03 * math.sin(step * 0.011)))
    aux_loss = max(0.001, loss * 0.11 + 0.02 * (1 - progress))
    perplexity = min(999.0, math.exp(min(val_loss, 6.9)))
    auc = min(0.999, accuracy + 0.025 + 0.015 * (1 - math.exp(-3 * progress)))
    precision = min(0.995, max(0.0, accuracy + 0.012 * math.sin(step * 0.013)))
    recall = min(0.995, max(0.0, accuracy - 0.018 * (1 - progress) + 0.01 * math.cos(step * 0.009)))
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    warmup_progress = min(1.0, progress / 0.08)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    learning_rate = 3e-4 * warmup_progress * max(0.08, cosine_decay)
    grad_norm = max(0.02, 1.8 * math.exp(-2.8 * progress) + 0.08 * math.sin(step * 0.019))
    samples_per_second = max(0.25, 8.8 + 1.4 * math.sin(step * 0.031) + rng.uniform(-0.55, 0.55))
    tokens_per_second = samples_per_second * (768 + 96 * math.sin(step * 0.014))
    requests_running = max(1, int(6 + 3 * math.sin(step * 0.021)))
    requests_waiting = max(0, int(4 + 4 * math.sin(step * 0.013 + 1.2)))
    gpu_cache_usage = min(0.96, max(0.35, 0.56 + 0.32 * progress + 0.04 * math.sin(step * 0.015)))
    prefix_cache_hit_rate = min(0.98, max(0.05, 0.22 + 0.64 * progress + 0.05 * math.cos(step * 0.017)))
    ttft_ms = max(18.0, 76.0 - 28.0 * progress + 5.0 * math.sin(step * 0.023))
    tpot_ms = max(4.0, 14.0 - 5.5 * progress + 1.2 * math.cos(step * 0.019))
    fwd_latency_ms = max(8.0, 66.0 - 18.0 * progress + 4.0 * math.sin(step * 0.027))
    bwd_latency_ms = max(12.0, 112.0 - 35.0 * progress + 5.5 * math.cos(step * 0.018))
    allreduce_latency_ms = max(1.4, 13.5 - 4.0 * progress + 1.1 * math.sin(step * 0.024))
    optimizer_step_ms = max(0.6, 5.8 - 1.4 * progress + 0.4 * math.cos(step * 0.016))
    gpu_memory_gb = min(79.0, 19.5 + 17.0 * progress + 1.1 * math.sin(step * 0.02))
    gpu_utilization = min(99.0, max(62.0, 88.0 + 7.0 * math.sin(step * 0.017)))
    loss_scale = 32768.0 if step < steps * 0.55 else 16384.0
    ema_decay = min(0.9999, 0.995 + 0.0045 * progress)
    diffusion_timestep = int(999 * (0.5 + 0.5 * math.sin(step * 0.041)))
    noise_offset = max(0.0, 0.08 + 0.02 * math.sin(step * 0.012))
    snr_gamma = 5.0

    metrics = TrainingMetrics(
        loss=loss,
        val_loss=val_loss,
        lm_loss=lm_loss,
        aux_loss=aux_loss,
        perplexity=perplexity,
        accuracy=accuracy,
        auc=auc,
        f1=min(0.995, f1),
        precision=precision,
        recall=recall,
        learning_rate=learning_rate,
        grad_norm=grad_norm,
        gpu_memory_gb=gpu_memory_gb,
        gpu_utilization=gpu_utilization,
        loss_scale=loss_scale,
        samples_per_second=samples_per_second,
        tokens_per_second=tokens_per_second,
        requests_running=requests_running,
        requests_waiting=requests_waiting,
        gpu_cache_usage=gpu_cache_usage,
        prefix_cache_hit_rate=prefix_cache_hit_rate,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        fwd_latency_ms=fwd_latency_ms,
        bwd_latency_ms=bwd_latency_ms,
        allreduce_latency_ms=allreduce_latency_ms,
        optimizer_step_ms=optimizer_step_ms,
        ema_decay=ema_decay,
        diffusion_timestep=diffusion_timestep,
        noise_offset=noise_offset,
        snr_gamma=snr_gamma,
    )
    return apply_scenario(metrics, scenario, progress, step)


def apply_scenario(metrics: TrainingMetrics, scenario: str, progress: float, step: int) -> TrainingMetrics:
    if scenario == "llm-pretrain":
        scale_wave = 1.0 + 0.05 * math.sin(step * 0.009)
        return replace(
            metrics,
            loss=max(0.02, metrics.loss * 1.08),
            val_loss=max(0.02, metrics.val_loss * 1.1),
            lm_loss=max(0.02, metrics.lm_loss * 1.14),
            aux_loss=max(0.001, metrics.aux_loss * 0.75),
            perplexity=min(999.0, metrics.perplexity * 1.16),
            grad_norm=max(0.03, metrics.grad_norm * 1.35),
            gpu_memory_gb=min(79.0, 42.0 + 27.0 * progress + 2.4 * math.sin(step * 0.013)),
            samples_per_second=max(0.2, metrics.samples_per_second * 0.34 * scale_wave),
            tokens_per_second=max(600.0, metrics.tokens_per_second * 7.5 * scale_wave),
            loss_scale=65536.0 if progress < 0.45 else metrics.loss_scale,
        )
    if scenario == "finetune":
        overfit_gap = max(0.0, progress - 0.62) * 0.26
        accuracy = min(0.995, metrics.accuracy + 0.025 * (1 - math.exp(-5 * progress)))
        precision = min(0.998, metrics.precision + 0.018)
        recall = min(0.998, metrics.recall + 0.012)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return replace(
            metrics,
            val_loss=metrics.val_loss + overfit_gap,
            accuracy=accuracy,
            auc=min(0.999, metrics.auc + 0.018),
            precision=precision,
            recall=recall,
            f1=min(0.998, f1),
            samples_per_second=max(0.4, metrics.samples_per_second * 0.78),
            tokens_per_second=max(250.0, metrics.tokens_per_second * 0.7),
            gpu_memory_gb=min(79.0, metrics.gpu_memory_gb + 3.0),
        )
    if scenario == "diffusion":
        return replace(
            metrics,
            loss=max(0.01, metrics.loss * (0.92 + 0.04 * math.sin(step * 0.021))),
            val_loss=max(0.01, metrics.val_loss * 0.95),
            samples_per_second=max(0.2, metrics.samples_per_second * 0.42),
            tokens_per_second=max(128.0, metrics.tokens_per_second * 0.25),
            gpu_memory_gb=min(79.0, 27.0 + 22.0 * progress + 1.9 * math.sin(step * 0.018)),
            ema_decay=min(0.99995, 0.996 + 0.0038 * progress),
            noise_offset=max(0.0, metrics.noise_offset + 0.015 * math.sin(step * 0.029)),
        )
    return metrics


def colorize(text: str, color_index: int, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{RAINBOW_COLORS[color_index % len(RAINBOW_COLORS)]}{text}{RESET_COLOR}"


def metric_segment(label: str, value: str, color_index: int, rainbow: bool) -> str:
    return f"{colorize(label, color_index, rainbow)}: {value}"


def format_trainer_log(
    step: int,
    steps: int,
    metrics: TrainingMetrics,
    rainbow: bool,
) -> str:
    epoch = step / steps
    segments = [
        f"'loss': {round(metrics.loss, 4)}",
        f"'grad_norm': {round(metrics.grad_norm, 4)}",
        f"'learning_rate': {metrics.learning_rate}",
        f"'epoch': {round(epoch, 2)}",
    ]
    return "{" + ", ".join(colorize(segment, index, rainbow) for index, segment in enumerate(segments)) + "}"


def format_deepspeed_log(step: int, metrics: TrainingMetrics, rainbow: bool) -> str:
    segments = [
        metric_segment("step", str(step), 0, rainbow),
        metric_segment("skipped", "0", 1, rainbow),
        metric_segment("lr", f"{metrics.learning_rate:.6e}", 2, rainbow),
        metric_segment("mom", "0.9000", 3, rainbow),
        metric_segment("loss_scale", f"{metrics.loss_scale:.0f}", 4, rainbow),
        metric_segment("samples/sec", f"{metrics.samples_per_second:.2f}", 5, rainbow),
        metric_segment("fwd", f"{metrics.fwd_latency_ms:.2f} ms", 6, rainbow),
        metric_segment("bwd", f"{metrics.bwd_latency_ms:.2f} ms", 7, rainbow),
        metric_segment("allreduce", f"{metrics.allreduce_latency_ms:.2f} ms", 8, rainbow),
        metric_segment("opt_step", f"{metrics.optimizer_step_ms:.2f} ms", 9, rainbow),
    ]
    return f"[rank0] DeepSpeed wall-clock | {' | '.join(segments)}"


def format_vllm_log(step: int, metrics: TrainingMetrics, rainbow: bool) -> str:
    segments = [
        metric_segment("Avg prompt throughput", f"{metrics.tokens_per_second * 0.42:.1f} tokens/s", 0, rainbow),
        metric_segment("Avg generation throughput", f"{metrics.tokens_per_second * 0.58:.1f} tokens/s", 1, rainbow),
        metric_segment("Running", str(metrics.requests_running), 2, rainbow),
        metric_segment("Waiting", str(metrics.requests_waiting), 3, rainbow),
        metric_segment("GPU KV cache usage", f"{metrics.gpu_cache_usage * 100:.1f}%", 4, rainbow),
        metric_segment("Prefix cache hit rate", f"{metrics.prefix_cache_hit_rate * 100:.1f}%", 5, rainbow),
        metric_segment("TTFT", f"{metrics.ttft_ms:.1f} ms", 6, rainbow),
        metric_segment("TPOT", f"{metrics.tpot_ms:.1f} ms", 7, rainbow),
    ]
    return f"INFO {step:>7} metrics.py: vLLM engine stats: {'; '.join(segments)}"


def format_stable_diffusion_log(step: int, steps: int, metrics: TrainingMetrics, rainbow: bool) -> str:
    epoch = step / steps
    segments = [
        metric_segment("step_loss", f"{metrics.loss:.4f}", 0, rainbow),
        metric_segment("lr", f"{metrics.learning_rate:.2e}", 1, rainbow),
        metric_segment("grad_norm", f"{metrics.grad_norm:.4f}", 2, rainbow),
        metric_segment("timestep", str(metrics.diffusion_timestep), 3, rainbow),
        metric_segment("ema_decay", f"{metrics.ema_decay:.5f}", 4, rainbow),
        metric_segment("snr_gamma", f"{metrics.snr_gamma:.1f}", 5, rainbow),
        metric_segment("noise_offset", f"{metrics.noise_offset:.3f}", 6, rainbow),
        metric_segment("gpu_mem", f"{metrics.gpu_memory_gb:.1f}GB", 7, rainbow),
        metric_segment("epoch", f"{epoch:.2f}", 8, rainbow),
    ]
    return f"Steps {step}/{steps} | {' | '.join(segments)}"


def format_log(style: str, step: int, steps: int, metrics: TrainingMetrics, rainbow: bool) -> str:
    if style == "deepspeed":
        return format_deepspeed_log(step, metrics, rainbow)
    if style == "vllm":
        return format_vllm_log(step, metrics, rainbow)
    if style == "stable-diffusion":
        return format_stable_diffusion_log(step, steps, metrics, rainbow)
    return format_trainer_log(step, steps, metrics, rainbow)


def format_event(level: str, message: str, rainbow: bool) -> str:
    color_index = {"INFO": 5, "WARNING": 1, "ERROR": 0}.get(level, 2)
    return f"{colorize(level, color_index, rainbow)} {message}"


def training_event(
    step: int,
    steps: int,
    scenario: str,
    chaos_level: int,
    metrics: TrainingMetrics,
    rng: random.Random,
    rainbow: bool = False,
) -> Optional[str]:
    if chaos_level == 0:
        return None

    probability = 0.025 if chaos_level == 1 else 0.2
    forced_small_demo = chaos_level == 2 and steps <= 5 and step in (2, steps)
    if not forced_small_demo and rng.random() > probability:
        return None

    common_events = [
        (
            "WARNING",
            f"trainer.py:{step}: dataloader worker heartbeat delayed; keeping batch queue warm",
        ),
        (
            "INFO",
            f"checkpointing.py:{step}: saving sharded optimizer state part {rng.randint(1, 8)}/8",
        ),
        (
            "WARNING",
            f"wandb: network timeout while syncing step {step}, retrying in {rng.randint(3, 9)}s",
        ),
    ]
    scenario_events = {
        "normal": [
            (
                "INFO",
                f"eval loop: val_loss={metrics.val_loss:.4f}, accuracy={metrics.accuracy:.4f}, best checkpoint unchanged",
            ),
            (
                "WARNING",
                f"scheduler: lr plateau detected near {metrics.learning_rate:.2e}; cosine decay continues",
            ),
        ],
        "llm-pretrain": [
            (
                "WARNING",
                f"cuda OOM at microbatch boundary; reducing activation checkpoint window, gpu_mem={metrics.gpu_memory_gb:.1f}GB",
            ),
            (
                "WARNING",
                f"fp16 overflow detected; skipping optimizer step and lowering loss_scale to {metrics.loss_scale / 2:.0f}",
            ),
            (
                "INFO",
                f"tokens/sec recovered to {metrics.tokens_per_second:.1f}; sequence packing efficiency stable",
            ),
        ],
        "finetune": [
            (
                "WARNING",
                f"eval_loss moved above train_loss by {metrics.val_loss - metrics.loss:.4f}; possible overfit smell",
            ),
            (
                "INFO",
                f"eval metrics jump: accuracy={metrics.accuracy:.4f}, f1={metrics.f1:.4f}; keeping adapter weights",
            ),
        ],
        "diffusion": [
            (
                "INFO",
                f"sample preview queued at diffusion timestep {metrics.diffusion_timestep}; EMA weights look stable",
            ),
            (
                "WARNING",
                f"latent cache miss burst; gpu_mem={metrics.gpu_memory_gb:.1f}GB, retrying dataloader prefetch",
            ),
        ],
    }
    dramatic_events = [
        (
            "ERROR",
            f"NaN detected in grad_norm={metrics.grad_norm:.4f}; skipped step {step} and restored previous scaler state",
        ),
        (
            "WARNING",
            f"loss spike observed: loss={metrics.loss + rng.uniform(0.4, 1.8):.4f}; gradient clipping engaged",
        ),
    ]

    event_pool = common_events + scenario_events.get(scenario, [])
    if chaos_level == 2:
        event_pool += dramatic_events
    level, message = rng.choice(event_pool)
    return format_event(level, message, rainbow)


def checkpoint_path(project_name: str, run_name: str, step: int) -> str:
    return str(PurePosixPath(project_name) / run_name / f"checkpoint-{step}" / "checkpoint.pth")


def simulated_step_delay(
    step: int,
    base_delay: float,
    speed_jitter: float,
    rng: Optional[random.Random] = None,
) -> float:
    if base_delay == 0:
        return 0
    rng = rng or random.Random()
    wave = 0.14 * math.sin(step * 0.037) + 0.08 * math.sin(step * 0.011)
    noise = rng.uniform(-speed_jitter, speed_jitter)
    multiplier = max(0.25, 1 + wave + noise)
    return base_delay * multiplier


def stage_for_step(spec: RunSpec, step: int) -> StageSpec:
    progress = (step - 1) / max(1, spec.steps - 1)
    previous = spec.stages[0]
    for stage in spec.stages:
        if stage.start_pct <= progress <= stage.end_pct:
            return stage
        if progress < stage.start_pct:
            return previous
        previous = stage
    return spec.stages[-1]


def validate_stage(stage: StageSpec) -> None:
    if stage.name not in STAGE_NAMES:
        raise ValueError(f"stage name must be one of: {', '.join(STAGE_NAMES)}")
    if not 0 <= stage.start_pct < stage.end_pct <= 1:
        raise ValueError(f"stage {stage.name!r} must satisfy 0 <= start_pct < end_pct <= 1")
    if stage.scenario not in SCENARIOS:
        raise ValueError(f"stage {stage.name!r} scenario must be one of: {', '.join(SCENARIOS)}")
    if stage.chaos_level not in CHAOS_LEVELS:
        raise ValueError(f"stage {stage.name!r} chaos_level must be 0, 1, or 2")
    for event in stage.events:
        if event.level not in {"INFO", "WARNING", "ERROR"}:
            raise ValueError(f"stage {stage.name!r} event level must be INFO, WARNING, or ERROR")
        if not stage.start_pct <= event.at_pct <= stage.end_pct:
            raise ValueError(f"stage {stage.name!r} event at_pct must be inside the stage range")


def validate_run_spec(spec: RunSpec) -> None:
    if not spec.name:
        raise ValueError("run name must not be empty")
    if spec.log_style not in LOG_STYLES:
        raise ValueError(f"log_style must be one of: {', '.join(LOG_STYLES)}")
    if spec.steps <= 0:
        raise ValueError("steps must be greater than 0")
    if spec.step_delay < 0:
        raise ValueError("step_delay must be non-negative")
    if spec.log_every <= 0:
        raise ValueError("log_every must be greater than 0")
    if spec.save_every < 0:
        raise ValueError("save_every must be non-negative")
    if spec.save_delay < 0:
        raise ValueError("save_delay must be non-negative")
    if spec.speed_jitter < 0:
        raise ValueError("speed_jitter must be non-negative")
    if spec.loss_start < spec.loss_min:
        raise ValueError("loss_start must be greater than or equal to loss_min")
    if not spec.project_name or not spec.run_name:
        raise ValueError("project_name and run_name must not be empty")
    if not spec.stages:
        raise ValueError("stages must contain at least one stage")
    last_end = 0.0
    for index, stage in enumerate(spec.stages):
        validate_stage(stage)
        if index == 0 and stage.start_pct != 0:
            raise ValueError("first stage must start at 0.0")
        if stage.start_pct < last_end:
            raise ValueError("stages must be sorted and must not overlap")
        last_end = stage.end_pct
    if spec.stages[-1].end_pct != 1:
        raise ValueError("last stage must end at 1.0")


def parse_incident(data: dict[str, Any], stage_name: str) -> IncidentSpec:
    allowed = {"level", "message", "at_pct"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"stage {stage_name!r} event has unsupported fields: {', '.join(sorted(unknown))}")
    return IncidentSpec(
        level=str(data.get("level", "INFO")),
        message=str(data.get("message", "")),
        at_pct=float(data.get("at_pct", 0)),
    )


def parse_stage(data: dict[str, Any]) -> StageSpec:
    allowed = {"name", "start_pct", "end_pct", "scenario", "chaos_level", "events"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"stage has unsupported fields: {', '.join(sorted(unknown))}")
    name = str(data.get("name", "train"))
    events = tuple(parse_incident(event, name) for event in data.get("events", []) or [])
    return StageSpec(
        name=name,
        start_pct=float(data.get("start_pct", 0)),
        end_pct=float(data.get("end_pct", 1)),
        scenario=str(data.get("scenario", "normal")),
        chaos_level=int(data.get("chaos_level", 1)),
        events=events,
    )


def load_run_spec(path: str) -> RunSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    allowed = {
        "name",
        "description",
        "kind",
        "log_style",
        "seed",
        "steps",
        "step_delay",
        "speed_jitter",
        "log_every",
        "save_every",
        "save_delay",
        "project_name",
        "run_name",
        "loss_start",
        "loss_min",
        "acc_start",
        "oscillation",
        "stages",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"config has unsupported fields: {', '.join(sorted(unknown))}")
    raw_stages = data.get("stages")
    stages = tuple(parse_stage(stage) for stage in raw_stages) if raw_stages else default_stages()
    spec = RunSpec(
        name=str(data.get("name", "custom-run")),
        description=str(data.get("description", "Custom TrainFake cinematic run.")),
        kind=str(data.get("kind", "custom")),
        log_style=str(data.get("log_style", "trainer")),
        seed=data.get("seed", 42),
        steps=int(data.get("steps", 240)),
        step_delay=float(data.get("step_delay", 0.03)),
        speed_jitter=float(data.get("speed_jitter", 0.18)),
        log_every=int(data.get("log_every", 12)),
        save_every=int(data.get("save_every", 80)),
        save_delay=float(data.get("save_delay", 0.15)),
        project_name=str(data.get("project_name", "trainfake")),
        run_name=str(data.get("run_name", "custom-run")),
        loss_start=float(data.get("loss_start", 1.5)),
        loss_min=float(data.get("loss_min", 0.1)),
        acc_start=float(data.get("acc_start", 0.5)),
        oscillation=float(data.get("oscillation", 0.05)),
        stages=stages,
    )
    validate_run_spec(spec)
    return spec


def resolve_run_spec(args: argparse.Namespace) -> RunSpec:
    if args.config and args.preset:
        raise ValueError("choose either a preset or --config, not both")
    if args.config:
        spec = load_run_spec(args.config)
    else:
        preset_name = args.preset or "llama-70b-pretrain"
        if preset_name not in PRESETS:
            raise ValueError(f"unknown preset {preset_name!r}; run `trainfake presets` to see available presets")
        spec = PRESETS[preset_name]
    overrides: dict[str, Any] = {}
    for name in ("steps", "step_delay", "seed", "log_every", "save_every"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if overrides:
        spec = replace(spec, **overrides)
    validate_run_spec(spec)
    return spec


def stage_banner(stage: StageSpec) -> str:
    labels = {
        "bootstrap": "environment scan",
        "warmup": "scheduler warmup",
        "train": "main training loop",
        "incident": "incident window",
        "recovery": "recovery and scaler stabilization",
        "eval": "evaluation pass",
        "checkpoint": "checkpoint consolidation",
        "summary": "run summary",
    }
    return labels.get(stage.name, stage.name)


def list_presets(console: Console) -> None:
    table = Table(title="TrainFake cinematic presets")
    table.add_column("preset", style="cyan", no_wrap=True)
    table.add_column("kind", style="magenta")
    table.add_column("steps", justify="right")
    table.add_column("log style")
    table.add_column("description")
    for preset in PRESETS.values():
        table.add_row(preset.name, preset.kind, str(preset.steps), preset.log_style, preset.description)
    console.print(table)


def render_summary(console: Console, summary: RunSummary) -> None:
    metrics = summary.final_metrics
    table = Table(title=f"summary: {summary.spec_name}")
    table.add_column("field", style="cyan")
    table.add_column("value", style="green")
    table.add_row("steps", str(summary.steps))
    table.add_row("final loss", "n/a" if metrics is None else f"{metrics.loss:.4f}")
    table.add_row("final grad_norm", "n/a" if metrics is None else f"{metrics.grad_norm:.4f}")
    table.add_row("best checkpoint", summary.best_checkpoint)
    table.add_row("incidents", str(summary.incidents))
    table.add_row("recoveries", str(summary.recoveries))
    table.add_row("simulated wall time", f"{summary.elapsed_seconds:.2f}s")
    console.print(table)


def run_cinematic(spec: RunSpec, console: Optional[Console] = None, rainbow: bool = False) -> RunSummary:
    validate_run_spec(spec)
    console = console or Console()
    rng = random.Random(spec.seed)
    summary = RunSummary(spec_name=spec.name, steps=spec.steps)
    emitted_scripted_events: set[tuple[str, float, str]] = set()
    current_stage_name: Optional[str] = None
    start_time = time.monotonic()

    console.rule(f"[bold cyan]TrainFake run: {spec.name}")
    console.print(f"[dim]{spec.description}[/dim]")
    console.print(
        f"INFO launcher.py: seed={spec.seed} log_style={spec.log_style} project={spec.project_name}/{spec.run_name}"
    )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task("training", total=spec.steps)
        for step in range(1, spec.steps + 1):
            stage = stage_for_step(spec, step)
            if stage.name != current_stage_name:
                current_stage_name = stage.name
                console.print(f"[bold blue]stage:{stage.name}[/bold blue] {stage_banner(stage)}")
                if stage.name == "recovery":
                    summary.recoveries += 1
                progress.update(task_id, description=stage.name)

            metrics = simulated_metrics(
                step=step,
                steps=spec.steps,
                loss_start=spec.loss_start,
                loss_min=spec.loss_min,
                acc_start=spec.acc_start,
                oscillation=spec.oscillation,
                scenario=stage.scenario,
                rng=rng,
            )
            summary.final_metrics = metrics
            time.sleep(simulated_step_delay(step, spec.step_delay, spec.speed_jitter, rng))

            if step == 1 or step % spec.log_every == 0 or step == spec.steps:
                console.print(format_log(spec.log_style, step, spec.steps, metrics, rainbow), markup=False)

            for event in stage.events:
                key = (stage.name, event.at_pct, event.message)
                step_progress = (step - 1) / max(1, spec.steps - 1)
                if key not in emitted_scripted_events and step_progress >= event.at_pct:
                    emitted_scripted_events.add(key)
                    summary.incidents += 1
                    console.print(format_event(event.level, event.message, rainbow), markup=False)

            event = training_event(step, spec.steps, stage.scenario, stage.chaos_level, metrics, rng, rainbow)
            if event:
                summary.incidents += 1
                console.print(event, markup=False)

            if spec.save_every and step % spec.save_every == 0:
                time.sleep(spec.save_delay)
                summary.best_checkpoint = checkpoint_path(spec.project_name, spec.run_name, step)
                console.print(f"INFO checkpointing.py: saved model checkpoint to {summary.best_checkpoint}")

            progress.advance(task_id)

    if summary.best_checkpoint == "none" and spec.save_every:
        summary.best_checkpoint = checkpoint_path(spec.project_name, spec.run_name, spec.steps)
    summary.elapsed_seconds = time.monotonic() - start_time
    render_summary(console, summary)
    return summary


def print_default_help(console: Console) -> None:
    console.print("[bold cyan]TrainFake[/bold cyan] is preset-first.")
    console.print("Try:")
    console.print("  trainfake presets")
    console.print("  trainfake run llama-70b-pretrain --step-delay 0")
    console.print("  trainfake run --config tests/fixtures/sample_run.yaml")


def print_legacy_hint(console: Console) -> None:
    console.print("[bold yellow]TrainFake uses a preset-first CLI.[/bold yellow]")
    console.print("Use `trainfake presets` or `trainfake run <preset>` instead of legacy top-level flags.")
    console.print("Example: `trainfake run bert-finetune --step-delay 0`")


def main() -> None:
    console = Console()
    if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
        print_legacy_hint(console)
        raise SystemExit(2)
    args = parse_args()
    try:
        if args.command == "presets":
            list_presets(console)
        elif args.command == "run":
            run_cinematic(resolve_run_spec(args), console, args.rainbow)
        else:
            print_default_help(console)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
