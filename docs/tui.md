# TUI Mode

The Textual-powered TUI keeps the same simulated training engine as `run`, but renders it as a terminal dashboard.

```bash
trainfake tui boss-is-watching
trainfake tui rag-eval-nightly --step-delay 0 --steps 50 --save-every 0
trainfake tui --config tests/fixtures/sample_run.yaml
```

## Layout

- Left pane: live log stream, stage changes, incidents, and checkpoint messages.
- Right pane: run status, progress, incident counts, checkpoint path, and ASCII metric curves.
- Default curves: `loss`, `val_loss`, `grad_norm`, `learning_rate`, `gpu_memory_gb`, and `samples_per_second`.
- Scenario-specific curves are added automatically, such as `reward_accuracy`, `faithfulness`, `contrastive_loss`, or `node_pressure`.

## Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `q` | Quit. |
| `p` | Pause or resume the simulation. |
| `l` | Focus the log pane. |
| `m` | Focus the metric pane. |
| `1`-`6` | Highlight a primary metric curve. |

## Notes

The TUI uses Textual and plain terminal-rendered curves. It does not require matplotlib, plotext, browser rendering, or a GPU.
