# TrainFake

[中文 README](README_zh.md)

![TrainFake hero](assets/readme-hero-v2.png)

> Cinematic AI training logs for demos, tests, teaching, and suspiciously serious terminals.

## Demo

![TrainFake demo](assets/demo.gif)

`TrainFake` is a preset-first terminal simulator for believable AI training runs. It does not train a real model, but it stages a complete deep learning session with environment scans, warmup, the main loop, incident windows, recovery logs, eval-flavored metrics, checkpoints, and a final summary table.

It is built for demos, log parser tests, teaching, screenshots, launch posts, and any moment when a terminal needs to look convincingly busy without touching a GPU. The CLI command is `trainfake`.

## Features

- Built-in scenarios for LLM pretraining, Diffusers LoRA training, vLLM launch rehearsals, BERT fine-tuning, RLHF reward models, RAG evals, multimodal pretraining, and GPU cluster drills.
- YAML run scripts for custom stages, log styles, scripted events, checkpoint cadence, and reproducible seeds.
- Hugging Face `Trainer`, DeepSpeed, vLLM, and Stable Diffusion/Diffusers-style log output.
- Reproducible metrics, optional colored log fields, zero-delay demos, and a final summary table.
- No real training, model downloads, or GPU usage.

## Install

Python 3.8 or newer is required.

```bash
python3 -m pip install .
```

For editable development installs:

```bash
python3 -m pip install -e .
```

## Quick Start

List the built-in presets:

```bash
trainfake presets
```

Run an LLM pretraining scene:

```bash
trainfake run llama-70b-pretrain
```

Run a fast demo with no real delay:

```bash
trainfake run boss-is-watching --step-delay 0
```

Run the split-screen TUI with live logs and metric curves:

```bash
trainfake tui boss-is-watching
```

Run from a YAML script:

```bash
trainfake run --config tests/fixtures/sample_run.yaml
```

Running `trainfake` with no subcommand prints recommended commands instead of starting a long default run.

## Built-in Presets

| Preset | Use case |
| --- | --- |
| `llama-70b-pretrain` | DeepSpeed-flavored LLM pretraining with warmup, loss scale, NCCL/allreduce texture, and sharded checkpoints. |
| `sdxl-lora` | Stable Diffusion/Diffusers-style LoRA training with EMA, diffusion timesteps, latent cache churn, and sample preview flavor. |
| `vllm-launch` | vLLM serving rehearsal with request queues, KV cache pressure, TTFT, TPOT, and throughput recovery. |
| `bert-finetune` | Classic supervised fine-tuning with eval metrics, accuracy/F1 movement, and mild overfit warnings. |
| `boss-is-watching` | A short but believable serious-terminal run. |
| `deadline-finetune` | A compact adapter fine-tune with deadline energy while keeping the logs realistic. |
| `rlhf-reward-model` | Reward model training with pairwise loss, chosen/rejected accuracy, and labeler queue pressure. |
| `rag-eval-nightly` | RAG evaluation with retrieval hit rate, context precision, faithfulness, and latency drift. |
| `multimodal-pretrain` | Vision-language pretraining with contrastive loss, image/text batches, and vision encoder cache churn. |
| `k8s-gpu-drill` | Kubernetes GPU rehearsal with pod scheduling, node pressure, NCCL probes, and checkpoint resume. |

## Commands

```bash
trainfake presets
trainfake run <preset>
trainfake run --config path/to/run.yaml
trainfake tui <preset>
trainfake tui --config path/to/run.yaml
```

`run` prints a classic streaming log. `tui` opens an interactive split-screen terminal view with live logs on the left and metric curves on the right.

`run` and `tui` support a small set of overrides:

| Flag | Description |
| --- | --- |
| `--steps` | Override the total number of simulated steps. |
| `--step-delay` | Override the base delay per step. Use `0` for demos. |
| `--seed` | Override the random seed. |
| `--log-every` | Override the metric logging cadence. |
| `--save-every` | Override checkpoint cadence. Use `0` to disable. |
| `--rainbow` | Enable ANSI colors inside log fields. |

Legacy top-level flags such as `--scenario`, `--log-style`, and `--steps` are no longer the primary interface. Use `trainfake run <preset>` or `trainfake run --config ...` instead.

## TUI Mode

The Textual-powered TUI keeps the same simulated training engine as `run`, but renders it as a terminal dashboard:

- Left pane: live log stream, stage changes, incidents, and checkpoint messages.
- Right pane: run status, progress, incident counts, checkpoint path, and ASCII metric curves.
- Default curves: `loss`, `val_loss`, `grad_norm`, `learning_rate`, `gpu_memory_gb`, and `samples_per_second`.
- Scenario-specific curves are added automatically, such as `reward_accuracy`, `faithfulness`, `contrastive_loss`, or `node_pressure`.

Keyboard shortcuts:

| Key | Action |
| --- | --- |
| `q` | Quit. |
| `p` | Pause or resume the simulation. |
| `l` | Focus the log pane. |
| `m` | Focus the metric pane. |
| `1`-`6` | Highlight a primary metric curve. |

## YAML Runs

A YAML script can define a custom training scene:

```yaml
name: office-demo
description: A compact cinematic run for a terminal demo.
log_style: deepspeed
seed: 42
steps: 120
step_delay: 0.02
log_every: 6
save_every: 30
project_name: demo
run_name: office-demo-rank0
stages:
  - name: bootstrap
    start_pct: 0.0
    end_pct: 0.1
    scenario: llm-pretrain
    chaos_level: 0
  - name: train
    start_pct: 0.1
    end_pct: 0.75
    scenario: llm-pretrain
    chaos_level: 1
  - name: incident
    start_pct: 0.75
    end_pct: 0.9
    scenario: llm-pretrain
    chaos_level: 2
    events:
      - level: WARNING
        message: "NCCL watchdog noticed a slow allreduce; rank0 keeps the job alive"
        at_pct: 0.8
  - name: summary
    start_pct: 0.9
    end_pct: 1.0
    scenario: llm-pretrain
    chaos_level: 0
```

Supported top-level fields include `name`, `description`, `log_style`, `seed`, `steps`, `step_delay`, `speed_jitter`, `log_every`, `save_every`, `save_delay`, `project_name`, `run_name`, `loss_start`, `loss_min`, `acc_start`, `oscillation`, and `stages`.

Each stage supports `name`, `start_pct`, `end_pct`, `scenario`, `chaos_level`, and `events`. Events support `level`, `message`, and `at_pct`.

You can mix the newer scenario types in YAML without changing the schema:

```yaml
name: rag-alignment-demo
description: RAG eval rolls into a compact reward-model pass.
log_style: trainer
steps: 80
step_delay: 0
save_every: 0
stages:
  - name: bootstrap
    start_pct: 0.0
    end_pct: 0.1
    scenario: rag-eval
    chaos_level: 0
  - name: eval
    start_pct: 0.1
    end_pct: 0.55
    scenario: rag-eval
    chaos_level: 1
    events:
      - level: INFO
        message: "faithfulness gate passed while retrieval_hit_rate stayed above target"
        at_pct: 0.4
  - name: train
    start_pct: 0.55
    end_pct: 0.95
    scenario: rlhf
    chaos_level: 1
  - name: summary
    start_pct: 0.95
    end_pct: 1.0
    scenario: rlhf
    chaos_level: 0
```

## Log Styles

- `trainer`: Hugging Face `Trainer`-style metric dictionaries.
- `deepspeed`: DeepSpeed wall-clock/profiler-style logs.
- `vllm`: vLLM serving metrics-style logs.
- `stable-diffusion`: Diffusers/Accelerate-style step logs.

Scenario-specific runs may add fields such as `reward_accuracy`, `faithfulness`, `contrastive_loss`, and `pod` status while keeping the same CLI and YAML shape.

## Stages

A cinematic run can include these stage names:

- `bootstrap`: startup, environment scan, seed, and project path details.
- `warmup`: learning-rate warmup and throughput ramp-up.
- `train`: main training loop.
- `incident`: OOM, overflow, NCCL, wandb retry, and loss spike windows.
- `recovery`: scaler, checkpoint, throughput, and queue recovery.
- `eval`: validation metrics and best-checkpoint decisions.
- `checkpoint`: checkpoint save or merge activity.
- `summary`: final table with loss, incidents, recoveries, and checkpoint details.

## Development

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run CLI smoke checks:

```bash
trainfake presets
trainfake run boss-is-watching --step-delay 0 --steps 5 --save-every 0
trainfake run rag-eval-nightly --step-delay 0 --steps 5 --save-every 0
```

## License

MIT
