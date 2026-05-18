# 使用指南

TrainFake 以 preset 为核心。最常见的使用方式是先查看内置场景，再运行其中一个。

```bash
trainfake presets
trainfake run llama-70b-pretrain
trainfake run boss-is-watching --step-delay 0
```

直接运行 `trainfake` 会显示推荐命令，不会自动启动一场很长的训练。

## 命令

```bash
trainfake presets
trainfake run <preset>
trainfake run --config path/to/run.yaml
trainfake tui <preset>
trainfake tui --config path/to/run.yaml
```

`run` 输出传统流式日志；`tui` 打开交互式左右分屏终端界面，左侧是实时日志，右侧是指标曲线。

## 覆盖参数

`run` 和 `tui` 支持同一组常用覆盖参数：

| 参数 | 说明 |
| --- | --- |
| `--steps` | 覆盖总模拟 step 数。 |
| `--step-delay` | 覆盖每 step 的基础等待时间，演示时可设为 `0`。 |
| `--seed` | 覆盖随机种子。 |
| `--log-every` | 覆盖指标日志频率。 |
| `--save-every` | 覆盖 checkpoint 频率，`0` 表示禁用。 |
| `--rainbow` | 为日志字段启用 ANSI 彩色输出。 |

旧版顶层参数如 `--scenario`、`--log-style`、`--steps` 不再是主入口。推荐改用 `trainfake run <preset>` 或 `trainfake run --config ...`。

## 内置 Preset

| Preset | 适合场景 |
| --- | --- |
| `llama-70b-pretrain` | DeepSpeed 风格的大模型预训练，包含 warmup、loss scale、NCCL/allreduce 氛围和 sharded checkpoint。 |
| `sdxl-lora` | Stable Diffusion/Diffusers 风格的 LoRA 训练，包含 EMA、diffusion timestep、latent cache 和 sample preview 味道。 |
| `vllm-launch` | vLLM serving 风格的上线彩排，突出请求队列、KV cache、TTFT、TPOT 和吞吐恢复。 |
| `bert-finetune` | 经典监督微调，突出 eval metrics、accuracy/F1 和轻微 overfit warning。 |
| `boss-is-watching` | 短平快但足够可信的“终端正在认真训练”场景。 |
| `deadline-finetune` | deadline 前的 adapter 微调现场，玩梗克制，日志仍保持真实感。 |
| `rlhf-reward-model` | 奖励模型训练现场，包含 pairwise loss、chosen/rejected accuracy 和 labeler queue 压力。 |
| `rag-eval-nightly` | RAG 评测流水线，包含 retrieval hit rate、context precision、faithfulness 和 latency drift。 |
| `multimodal-pretrain` | 多模态预训练现场，包含 contrastive loss、image/text batch 和 vision encoder cache。 |
| `k8s-gpu-drill` | Kubernetes GPU 演练，包含 pod 调度、node pressure、NCCL 探测和 checkpoint resume。 |

## 日志风格

- `trainer`：Hugging Face `Trainer` 风格指标字典。
- `deepspeed`：DeepSpeed wall-clock/profiler 风格日志。
- `vllm`：vLLM serving metrics 风格日志。
- `stable-diffusion`：Diffusers/Accelerate 风格 step 日志。

新场景会在保持 CLI 和 YAML 结构不变的基础上输出 `reward_accuracy`、`faithfulness`、`contrastive_loss`、`pod` 状态等特色字段。
