from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import DefaultDict, Iterable, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, RichLog, Static

from .main import RunFrame, RunSpec, TrainingMetrics, iter_run_frames, stage_banner


PRIMARY_METRICS = (
    "loss",
    "val_loss",
    "grad_norm",
    "learning_rate",
    "gpu_memory_gb",
    "samples_per_second",
)

SCENARIO_METRICS = {
    "rlhf": ("reward_accuracy", "pairwise_loss", "chosen_margin"),
    "rag-eval": ("faithfulness", "retrieval_hit_rate", "context_precision", "p95_latency_ms"),
    "multimodal": ("contrastive_loss", "vision_cache_hit_rate", "image_batch"),
    "k8s-gpu": ("node_pressure", "pods_running", "checkpoint_resume_count"),
    "llm-pretrain": ("tokens_per_second", "loss_scale", "allreduce_latency_ms"),
    "diffusion": ("ema_decay", "diffusion_timestep", "noise_offset"),
    "finetune": ("accuracy", "f1", "auc"),
}

SPARK_CHARS = " .:-=+*#%@"


def metric_value(metrics: TrainingMetrics, name: str) -> float:
    value = getattr(metrics, name)
    return float(value)


def format_metric(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def sparkline(values: Iterable[float], width: int) -> str:
    points = list(values)[-width:]
    if not points:
        return ""
    low = min(points)
    high = max(points)
    if high == low:
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(points)
    scale = len(SPARK_CHARS) - 1
    return "".join(SPARK_CHARS[int((value - low) / (high - low) * scale)] for value in points)


class StatusPanel(Static):
    def update_frame(self, spec: RunSpec, frame: RunFrame, paused: bool) -> None:
        summary = frame.summary_state
        status = "paused" if paused else ("done" if frame.final else "running")
        checkpoint = summary.best_checkpoint
        text = (
            f"{spec.name}\n"
            f"status={status} stage={frame.stage.name} step={frame.step}/{spec.steps}\n"
            f"elapsed={summary.elapsed_seconds:.2f}s incidents={summary.incidents} recoveries={summary.recoveries}\n"
            f"checkpoint={checkpoint}"
        )
        self.update(text)


class MetricChart(Static):
    can_focus = True

    def __init__(self, history_limit: int = 80, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history_limit = history_limit
        self.histories: DefaultDict[str, list[float]] = defaultdict(list)
        self.metric_names = list(PRIMARY_METRICS)
        self.selected_metric: Optional[int] = None

    def update_metrics(self, frame: RunFrame) -> None:
        if frame.metrics is None:
            return
        scenario_names = SCENARIO_METRICS.get(frame.stage.scenario, ())
        self.metric_names = list(dict.fromkeys(PRIMARY_METRICS + scenario_names))
        for name in self.metric_names:
            values = self.histories[name]
            values.append(metric_value(frame.metrics, name))
            del values[:-self.history_limit]
        self.refresh_chart()

    def set_selected_metric(self, index: Optional[int]) -> None:
        self.selected_metric = index
        self.refresh_chart()

    def refresh_chart(self) -> None:
        width = max(12, min(self.history_limit, self.size.width - 34 if self.size.width else 48))
        lines = ["metrics"]
        for index, name in enumerate(self.metric_names, start=1):
            values = self.histories.get(name, [])
            if not values:
                continue
            current = values[-1]
            low = min(values)
            high = max(values)
            previous = values[-2] if len(values) > 1 else current
            trend = "+" if current > previous else "-" if current < previous else "="
            prefix = ">" if self.selected_metric == index else " "
            line = (
                f"{prefix}{index:>2} {name:<22} {format_metric(current):>10} {trend} "
                f"[{format_metric(low)}..{format_metric(high)}] {sparkline(values, width)}"
            )
            lines.append(line)
        self.update("\n".join(lines))


class TrainFakeTui(App):
    CSS = """
    Screen {
        background: #0d1117;
    }

    #main {
        height: 1fr;
    }

    #logs {
        width: 52%;
        height: 100%;
        border: solid #2f81f7;
        padding: 0 1;
    }

    #right {
        width: 48%;
        height: 100%;
    }

    #status {
        height: 5;
        border: solid #3fb950;
        padding: 0 1;
    }

    #metrics {
        height: 1fr;
        border: solid #d29922;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "toggle_pause", "Pause"),
        ("l", "focus_log", "Logs"),
        ("m", "focus_metrics", "Metrics"),
        ("1", "select_metric(1)", "Metric 1"),
        ("2", "select_metric(2)", "Metric 2"),
        ("3", "select_metric(3)", "Metric 3"),
        ("4", "select_metric(4)", "Metric 4"),
        ("5", "select_metric(5)", "Metric 5"),
        ("6", "select_metric(6)", "Metric 6"),
    ]

    def __init__(self, spec: RunSpec, rainbow: bool = False) -> None:
        super().__init__()
        self.spec = spec
        self.rainbow = rainbow
        self.paused = False
        self.finished = False
        self.stop_requested = False
        self.worker_thread: Optional[threading.Thread] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield RichLog(id="logs", highlight=False, markup=False, wrap=True)
            with Vertical(id="right"):
                yield StatusPanel(id="status")
                yield MetricChart(id="metrics")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#logs", RichLog)
        log.write(f"TrainFake TUI run: {self.spec.name}")
        log.write(self.spec.description)
        log.write(
            f"INFO launcher.py: seed={self.spec.seed} log_style={self.spec.log_style} "
            f"project={self.spec.project_name}/{self.spec.run_name}"
        )
        first_stage = self.spec.stages[0]
        initial_frame = RunFrame(
            step=0,
            stage=first_stage,
            metrics=None,
            summary_state=self._empty_summary(),
        )
        self.query_one("#status", StatusPanel).update_frame(self.spec, initial_frame, self.paused)
        self.worker_thread = threading.Thread(target=self._stream_frames, daemon=True)
        self.worker_thread.start()

    def _empty_summary(self):
        from .main import RunSummary

        return RunSummary(spec_name=self.spec.name, steps=self.spec.steps)

    def _stream_frames(self) -> None:
        for frame in iter_run_frames(self.spec, self.rainbow, sleep=False):
            if self.stop_requested:
                return
            self._sleep_with_pause(frame.delay_seconds)
            if self.stop_requested:
                return
            self.call_from_thread(self.apply_frame, frame)
            if frame.final:
                return

    def _sleep_with_pause(self, seconds: float) -> None:
        remaining = max(0.0, seconds)
        while remaining > 0 and not self.stop_requested:
            if self.paused:
                time.sleep(0.05)
                continue
            chunk = min(0.05, remaining)
            time.sleep(chunk)
            remaining -= chunk
        while self.paused and not self.stop_requested:
            time.sleep(0.05)

    def apply_frame(self, frame: RunFrame) -> None:
        log = self.query_one("#logs", RichLog)
        status = self.query_one("#status", StatusPanel)
        chart = self.query_one("#metrics", MetricChart)

        status.update_frame(self.spec, frame, self.paused)
        if frame.final:
            self.finished = True
            log.write(f"summary: {self.spec.name}")
            log.write(
                f"steps={frame.summary_state.steps} incidents={frame.summary_state.incidents} "
                f"recoveries={frame.summary_state.recoveries} elapsed={frame.summary_state.elapsed_seconds:.2f}s"
            )
            return

        if frame.stage_started:
            log.write(f"stage:{frame.stage.name} {stage_banner(frame.stage)}")
        if frame.log_line:
            log.write(frame.log_line)
        for event in frame.events:
            log.write(event)
        if frame.checkpoint_path:
            log.write(f"INFO checkpointing.py: saved model checkpoint to {frame.checkpoint_path}")
        chart.update_metrics(frame)

    def action_toggle_pause(self) -> None:
        if self.finished:
            return
        self.paused = not self.paused
        chart = self.query_one("#metrics", MetricChart)
        chart.refresh_chart()

    def action_focus_log(self) -> None:
        self.query_one("#logs", RichLog).focus()

    def action_focus_metrics(self) -> None:
        self.query_one("#metrics", MetricChart).focus()

    def action_select_metric(self, index: int) -> None:
        chart = self.query_one("#metrics", MetricChart)
        chart.set_selected_metric(index)
        chart.focus()

    def on_unmount(self) -> None:
        self.stop_requested = True


def run_tui(spec: RunSpec, rainbow: bool = False) -> None:
    TrainFakeTui(spec, rainbow).run()
