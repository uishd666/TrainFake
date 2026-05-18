# Development

For editable development installs:

```bash
python3 -m pip install -e .
```

Run tests:

```bash
python3 -m unittest discover -s tests
```

Run CLI smoke checks:

```bash
trainfake presets
trainfake run boss-is-watching --step-delay 0 --steps 5 --save-every 0
trainfake run rag-eval-nightly --step-delay 0 --steps 5 --save-every 0
```

Run a quick TUI check:

```bash
trainfake tui boss-is-watching --step-delay 0 --steps 5 --save-every 0
```
