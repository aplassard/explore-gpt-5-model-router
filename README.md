# explore-gpt-5-model-router
Let’s try and understand what the model router is doing

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency
management. The included `.env` file must contain a valid
`OPENAI_API_KEY`.

Install dependencies and create the virtual environment:

```bash
uv sync
```

## Running the probe

The repository includes a small script, `router_probe.py`, which sends a
handful of prompts to the `gpt-5` model alias and records the router's
fingerprint for each. The script supports concurrent API calls using a
thread pool via the `--threads` option and displays a progress bar for
each experiment using `tqdm`.

```bash
uv run python router_probe.py --runs 5 --threads 5 --temperature 1 --out-dir results
```

Results will be written to the `results/` directory and a summary will
be printed to the console. Each experiment produces its own CSV file as
well as a `combined_results.csv`, `summary.json`, and a human-readable
`report.md`.
