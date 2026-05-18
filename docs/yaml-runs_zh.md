# YAML 剧本

YAML 剧本可以定义一次自定义训练现场：

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

运行方式：

```bash
trainfake run --config path/to/run.yaml
trainfake tui --config path/to/run.yaml
```

## 顶层字段

支持的顶层字段包括 `name`、`description`、`kind`、`log_style`、`seed`、`steps`、`step_delay`、`speed_jitter`、`log_every`、`save_every`、`save_delay`、`project_name`、`run_name`、`loss_start`、`loss_min`、`acc_start`、`oscillation` 和 `stages`。

## 阶段和事件

`stages` 支持 `name`、`start_pct`、`end_pct`、`scenario`、`chaos_level`、`events`。事件支持 `level`、`message` 和 `at_pct`。

阶段名称：

- `bootstrap`：启动、环境扫描、seed 和项目路径信息。
- `warmup`：学习率 warmup、吞吐爬升。
- `train`：主训练循环。
- `incident`：OOM、overflow、NCCL、wandb retry、loss spike 等事故窗口。
- `recovery`：scaler、checkpoint、吞吐和队列恢复。
- `eval`：验证集指标和 best checkpoint 判断。
- `checkpoint`：checkpoint 合并或保存。
- `summary`：运行结束表格，展示最终 loss、事故次数、恢复次数和 checkpoint。

## 混合场景示例

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
