# TrainFake

[中文 README](README_zh.md) | [Documentation index](docs/README.md)

![TrainFake hero](assets/readme-hero-v2.png)

> Cinematic AI training logs for demos, tests, teaching, and suspiciously serious terminals.

`TrainFake` is a preset-first terminal simulator for believable AI training runs. It does not train a real model, download weights, or touch a GPU. It stages the terminal experience around environment scans, warmup, training loops, incidents, recovery logs, eval-like metrics, checkpoints, and summaries.

## Quick Start

Python 3.8 or newer is required.

```bash
python3 -m pip install .
trainfake presets
trainfake run boss-is-watching --step-delay 0
trainfake tui boss-is-watching
```

Run a custom YAML script:

```bash
trainfake run --config tests/fixtures/sample_run.yaml
```

## What It Includes

- Built-in scenes for LLM pretraining, Diffusers LoRA, vLLM launch rehearsals, BERT fine-tuning, RLHF reward models, RAG evals, multimodal pretraining, and GPU cluster drills.
- Classic streaming logs with `trainfake run`.
- Split-screen Textual dashboard with `trainfake tui`.
- YAML run scripts for custom stages, events, log styles, checkpoint cadence, and reproducible seeds.

## Docs

- [Usage guide](docs/usage.md)
- [TUI mode](docs/tui.md)
- [YAML run scripts](docs/yaml-runs.md)
- [Development](docs/development.md)

## Demo

![TrainFake demo](assets/demo.gif)

## License

MIT
