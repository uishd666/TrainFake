# slackDL

[English README](README.md)

一个为深度学习从业者、实验室选手和临时需要“电脑正在努力训练”的朋友准备的小工具。它不会帮你把 loss 真正降下去，但会认真地打印训练日志、滚动进度条、模拟保存 checkpoint，让屏幕看起来像是在跑一场很有前途的大实验。适合演示、测试日志解析，也适合在需要合理摸鱼的时候，给终端一点忙碌的尊严。

`slackDL` 是一个小型 Python CLI 包，用来模拟深度学习训练日志。默认情况下，它可以打印 Hugging Face Transformers `Trainer` 风格的字典日志，也可以模拟 DeepSpeed、vLLM、Stable Diffusion/Diffusers 风格的指标输出，并通过 `tqdm` 进度条展示不断变化的 `sample/s` 吞吐量。

## 安装

```bash
python3 -m pip install .
slackDL
```

## 使用

```bash
slackDL --steps 27161 --loss-start 1.5 --loss-min 0.1 --acc-start 0.5 --oscillation 0.05 --step-delay 0.12 --log-every 100 --save-every 1900 --project-name project-name --run-name run-name
```

参数：

- `--steps`：要模拟的训练样本数，默认 `27161`
- `--loss-start`：初始 loss，默认 `1.5`
- `--loss-min`：进入震荡前的最小 loss，默认 `0.1`
- `--acc-start`：初始 accuracy，默认 `0.5`
- `--oscillation`：收敛后的 loss 震荡幅度，默认 `0.05`
- `--step-delay`：每个样本的基础等待秒数，默认 `0.12`
- `--speed-jitter`：每个样本的相对速度抖动，默认 `0.22`
- `--log-every`：每 N 个样本打印一次日志，默认 `100`
- `--log-style`：选择输出风格，可选 `trainer`、`deepspeed`、`vllm` 或 `stable-diffusion`；默认 `trainer`
- `--save-every`：每 N 个样本模拟保存一次 checkpoint，默认 `1900`；使用 `0` 可禁用
- `--save-delay`：模拟保存 checkpoint 时暂停的秒数，默认 `1.2`
- `--project-name`：模拟 checkpoint 路径中的项目目录，默认 `project-name`
- `--run-name`：模拟 checkpoint 路径中的运行目录，默认 `run-name`
- `--rainbow`：为进度条和日志字段启用彩色 ANSI 输出

为了兼容旧用法，`--epochs` 和 `--epoch-delay` 仍然可以作为 `--steps` 和 `--step-delay` 的别名使用。

日志风格：

- `trainer`：Hugging Face `Trainer` 风格的指标字典，包含 `loss`、`grad_norm`、`learning_rate` 和 `epoch`。
- `deepspeed`：带有 wall-clock/profiler 味道的输出，包括跳过的优化器更新、momentum、fp16 `loss_scale`、samples/sec、forward/backward/allreduce 耗时，以及优化器 step 延迟。
- `vllm`：受 vLLM 已记录 engine stats 和 Prometheus 指标启发的 serving 指标，包括 prompt/generation token 吞吐量、running/waiting requests、KV-cache 使用率、prefix-cache 命中率、TTFT 和 TPOT。vLLM 主要是推理/服务框架，因此这个风格刻意偏 serving，而不是 optimizer step。
- `stable-diffusion`：Diffusers/Accelerate 风格的 step 输出，包括 `step_loss`、lr、grad norm、采样 diffusion timestep、EMA decay、SNR gamma、noise offset、GPU memory 和 epoch。

示例：

```bash
slackDL --log-style deepspeed --rainbow
slackDL --log-style vllm
slackDL --log-style stable-diffusion --log-every 50
```

示例输出：

```text
 12%|█████████████████▉                        | 3236/27161 [06:27<46:47,  8.52sample/s]
{'loss': 1.1071, 'grad_norm': 0.9592, 'learning_rate': 0.00025606601717798213, 'epoch': 0.25}
{'loss': 0.6365, 'grad_norm': 0.5202, 'learning_rate': 0.00015, 'epoch': 0.5}
Saving model checkpoint to project-name/run-name/checkpoint-1900/checkpoint.pth
{'loss': 0.1559, 'grad_norm': 0.2431, 'learning_rate': 4.3933982822017885e-05, 'epoch': 0.75}
```

## 许可证

MIT
