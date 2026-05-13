# slackDL

[Chinese README](README_zh.md)

![slackDL hero](assets/readme-hero.png)

A little tool for deep learning professionals, lab enthusiasts, and anyone who needs to make it look like their computer is “hard at work training” for a moment. It won’t actually help you reduce your loss, but it will diligently print training logs, display a scrolling progress bar, and simulate checkpoint saves—making your screen look like it’s running a promising large-scale experiment. It’s perfect for demos, parsing test logs, or simply giving your terminal a bit of “busy dignity” when you need a reasonable excuse to slack off.

`slackDL` is a small Python CLI package that simulates deep learning training logs. It can print Hugging Face Transformers `Trainer` dictionaries by default, or mimic DeepSpeed, vLLM, and Stable Diffusion/Diffusers-style metric output while showing a `tqdm` progress bar with changing `sample/s` throughput.

## Install

```bash
python3 -m pip install .
slackDL
```

## Usage

```bash
slackDL --steps 27161 --loss-start 1.5 --loss-min 0.1 --acc-start 0.5 --oscillation 0.05 --step-delay 0.12 --log-every 100 --save-every 100 --project-name project-name --run-name run-name
```

Options:

- `--steps`: number of training samples to simulate, default `27161`
- `--loss-start`: initial loss, default `1.5`
- `--loss-min`: minimum loss before oscillation, default `0.1`
- `--acc-start`: initial accuracy, default `0.5`
- `--oscillation`: loss oscillation amplitude after convergence, default `0.05`
- `--step-delay`: base seconds to wait between samples, default `0.12`
- `--speed-jitter`: relative per-sample speed variation, default `0.22`
- `--log-every`: print logs every N samples, default `100`
- `--log-style`: choose the output style, one of `trainer`, `deepspeed`, `vllm`, or `stable-diffusion`; default `trainer`
- `--save-every`: simulate saving a checkpoint every N samples, default `1900`; use `0` to disable
- `--save-delay`: seconds to pause when simulating checkpoint saves, default `1.2`
- `--project-name`: project directory used in simulated checkpoint paths, default `project-name`
- `--run-name`: run directory used in simulated checkpoint paths, default `run-name`
- `--rainbow`: enable colored ANSI output for the progress bar and log fields

For compatibility, `--epochs` and `--epoch-delay` still work as aliases for `--steps` and `--step-delay`.

Log styles:

- `trainer`: Hugging Face `Trainer`-style metric dictionary with `loss`, `grad_norm`, `learning_rate`, and `epoch`.
- `deepspeed`: wall-clock/profiler flavored output with skipped optimizer updates, momentum, fp16 `loss_scale`, samples/sec, forward/backward/allreduce timing, and optimizer step latency.
- `vllm`: vLLM serving metrics inspired by its logged engine stats and Prometheus metrics: prompt/generation token throughput, running/waiting requests, KV-cache usage, prefix-cache hit rate, TTFT, and TPOT. vLLM is primarily an inference/serving framework, so this style is intentionally serving-like rather than optimizer-step-like.
- `stable-diffusion`: Diffusers/Accelerate-style step output with `step_loss`, lr, grad norm, sampled diffusion timestep, EMA decay, SNR gamma, noise offset, GPU memory, and epoch.

Examples:

```bash
slackDL --log-style deepspeed --rainbow
slackDL --log-style vllm
slackDL --log-style stable-diffusion --log-every 50
```

Example output:

```text
 12%|█████████████████▉                        | 3236/27161 [06:27<46:47,  8.52sample/s]
{'loss': 1.1071, 'grad_norm': 0.9592, 'learning_rate': 0.00025606601717798213, 'epoch': 0.25}
{'loss': 0.6365, 'grad_norm': 0.5202, 'learning_rate': 0.00015, 'epoch': 0.5}
Saving model checkpoint to project-name/run-name/checkpoint-1900/checkpoint.pth
{'loss': 0.1559, 'grad_norm': 0.2431, 'learning_rate': 4.3933982822017885e-05, 'epoch': 0.75}
```

## License

MIT
