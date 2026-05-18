# 开发说明

开发模式安装：

```bash
python3 -m pip install -e .
```

运行测试：

```bash
python3 -m unittest discover -s tests
```

CLI smoke test：

```bash
trainfake presets
trainfake run boss-is-watching --step-delay 0 --steps 5 --save-every 0
trainfake run rag-eval-nightly --step-delay 0 --steps 5 --save-every 0
```

快速检查 TUI：

```bash
trainfake tui boss-is-watching --step-delay 0 --steps 5 --save-every 0
```
