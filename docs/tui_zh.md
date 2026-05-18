# TUI 模式

Textual 驱动的 TUI 使用与 `run` 相同的训练模拟逻辑，但会渲染成一个终端仪表盘。

```bash
trainfake tui boss-is-watching
trainfake tui rag-eval-nightly --step-delay 0 --steps 50 --save-every 0
trainfake tui --config tests/fixtures/sample_run.yaml
```

## 布局

- 左侧：实时日志流、阶段切换、事故事件和 checkpoint 消息。
- 右侧：运行状态、进度、事故计数、checkpoint 路径和 ASCII 指标曲线。
- 默认曲线：`loss`、`val_loss`、`grad_norm`、`learning_rate`、`gpu_memory_gb`、`samples_per_second`。
- 根据场景自动追加特色指标，例如 `reward_accuracy`、`faithfulness`、`contrastive_loss`、`node_pressure`。

## 快捷键

| 按键 | 动作 |
| --- | --- |
| `q` | 退出。 |
| `p` | 暂停或继续模拟。 |
| `l` | 聚焦日志面板。 |
| `m` | 聚焦指标面板。 |
| `1`-`6` | 高亮一个主要指标曲线。 |

## 说明

TUI 使用 Textual 和终端文本曲线，不需要 matplotlib、plotext、浏览器渲染或 GPU。
