import argparse
import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Optional

from tqdm import tqdm


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slackDL",
        description="Simulate deep learning training logs with realistic training incidents.",
    )
    parser.add_argument("--steps", type=int, default=27161, help="Number of training samples to simulate.")
    parser.add_argument("--epochs", dest="steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loss-start",
        type=float,
        default=1.5,
        help="Initial loss value.",
    )
    parser.add_argument(
        "--loss-min",
        type=float,
        default=0.1,
        help="Minimum loss value before oscillation.",
    )
    parser.add_argument(
        "--acc-start",
        type=float,
        default=0.5,
        help="Initial accuracy value.",
    )
    parser.add_argument(
        "--oscillation",
        type=float,
        default=0.05,
        help="Amplitude of post-convergence loss oscillation.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.12,
        help="Base seconds to wait between samples, default 0.12.",
    )
    parser.add_argument("--epoch-delay", dest="step_delay", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print logs every N samples, default 100.",
    )
    parser.add_argument(
        "--log-style",
        choices=LOG_STYLES,
        default="trainer",
        help="Choose a classic framework log style: trainer, deepspeed, vllm, or stable-diffusion.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="normal",
        help="Choose a training storyline: normal, llm-pretrain, finetune, or diffusion.",
    )
    parser.add_argument(
        "--chaos-level",
        type=int,
        choices=CHAOS_LEVELS,
        default=1,
        help="Density of realistic training incidents: 0 disables them, 1 is subtle, 2 is dramatic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed the simulated metrics and incidents for reproducible demos.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1900,
        help="Simulate saving a checkpoint every N samples. Use 0 to disable, default 1900.",
    )
    parser.add_argument(
        "--save-delay",
        type=float,
        default=1.2,
        help="Seconds to pause when simulating checkpoint saves, default 1.2.",
    )
    parser.add_argument(
        "--project-name",
        default="project-name",
        help="Project directory used in simulated checkpoint paths.",
    )
    parser.add_argument(
        "--run-name",
        default="run-name",
        help="Run directory used in simulated checkpoint paths.",
    )
    parser.add_argument(
        "--speed-jitter",
        type=float,
        default=0.22,
        help="Relative per-sample speed variation, default 0.22.",
    )
    parser.add_argument(
        "--rainbow",
        action="store_true",
        help="Enable colored ANSI output for the progress bar.",
    )
    return parser.parse_args()


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


def run_training(args: argparse.Namespace) -> None:
    validate_args(args)
    rng = random.Random(args.seed)

    progress_bar = tqdm(
        range(1, args.steps + 1),
        unit="sample",
        dynamic_ncols=True,
        smoothing=0.15,
        colour="cyan" if args.rainbow else None,
    )
    for step in progress_bar:
        metrics = simulated_metrics(
            step=step,
            steps=args.steps,
            loss_start=args.loss_start,
            loss_min=args.loss_min,
            acc_start=args.acc_start,
            oscillation=args.oscillation,
            scenario=args.scenario,
            rng=rng,
        )
        time.sleep(simulated_step_delay(step, args.step_delay, args.speed_jitter, rng))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            tqdm.write(format_log(args.log_style, step, args.steps, metrics, args.rainbow))
        event = training_event(step, args.steps, args.scenario, args.chaos_level, metrics, rng, args.rainbow)
        if event:
            tqdm.write(event)
        if args.save_every and step % args.save_every == 0:
            time.sleep(args.save_delay)
            tqdm.write(f"Saving model checkpoint to {checkpoint_path(args.project_name, args.run_name, step)}")


def main() -> None:
    args = parse_args()
    try:
        run_training(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
