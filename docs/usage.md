# Usage Guide

TrainFake is built around presets. The fastest way to start is to list the available scenes, then run one of them.

```bash
trainfake presets
trainfake run llama-70b-pretrain
trainfake run boss-is-watching --step-delay 0
```

Running `trainfake` with no subcommand prints recommended commands instead of starting a long default run.

## Commands

```bash
trainfake presets
trainfake run <preset>
trainfake run --config path/to/run.yaml
trainfake tui <preset>
trainfake tui --config path/to/run.yaml
```

`run` prints classic streaming logs. `tui` opens an interactive split-screen terminal view with live logs on the left and metric curves on the right.

## Overrides

`run` and `tui` support the same common overrides:

| Flag | Description |
| --- | --- |
| `--steps` | Override the total number of simulated steps. |
| `--step-delay` | Override the base delay per step. Use `0` for demos. |
| `--seed` | Override the random seed. |
| `--log-every` | Override metric logging cadence. |
| `--save-every` | Override checkpoint cadence. Use `0` to disable. |
| `--rainbow` | Enable ANSI colors inside log fields. |

Legacy top-level flags such as `--scenario`, `--log-style`, and `--steps` are no longer the primary interface. Use `trainfake run <preset>` or `trainfake run --config ...` instead.

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

## Log Styles

- `trainer`: Hugging Face `Trainer`-style metric dictionaries.
- `deepspeed`: DeepSpeed wall-clock/profiler-style logs.
- `vllm`: vLLM serving metrics-style logs.
- `stable-diffusion`: Diffusers/Accelerate-style step logs.

Scenario-specific runs may add fields such as `reward_accuracy`, `faithfulness`, `contrastive_loss`, and `pod` status while keeping the same CLI and YAML shape.
