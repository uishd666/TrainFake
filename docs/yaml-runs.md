# YAML Run Scripts

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

Run it with:

```bash
trainfake run --config path/to/run.yaml
trainfake tui --config path/to/run.yaml
```

## Top-level Fields

Supported top-level fields include `name`, `description`, `kind`, `log_style`, `seed`, `steps`, `step_delay`, `speed_jitter`, `log_every`, `save_every`, `save_delay`, `project_name`, `run_name`, `loss_start`, `loss_min`, `acc_start`, `oscillation`, and `stages`.

## Stages and Events

Each stage supports `name`, `start_pct`, `end_pct`, `scenario`, `chaos_level`, and `events`. Events support `level`, `message`, and `at_pct`.

Stage names:

- `bootstrap`: startup, environment scan, seed, and project path details.
- `warmup`: learning-rate warmup and throughput ramp-up.
- `train`: main training loop.
- `incident`: OOM, overflow, NCCL, wandb retry, and loss spike windows.
- `recovery`: scaler, checkpoint, throughput, and queue recovery.
- `eval`: validation metrics and best-checkpoint decisions.
- `checkpoint`: checkpoint save or merge activity.
- `summary`: final table with loss, incidents, recoveries, and checkpoint details.

## Mixed Scenario Example

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
