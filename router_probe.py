#! /usr/bin/env python3
"""Simple router probe for GPT-5 model router.

This script runs a small suite of prompts designed to trigger
different paths inside the GPT-5 router and records the
``system_fingerprint`` returned by the API for each prompt.

Results are written to ``results`` with one CSV per experiment as well
as a combined CSV and a summary JSON mapping experiments to router
fingerprints and counts.  The script prints a human readable summary to
stdout as well.
"""

import os
import time
import csv
import json
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

# Load variables from .env (OPENAI_API_KEY)
load_dotenv()

try:
    from openai import OpenAI
    HAS_SDK = True
except Exception:
    HAS_SDK = False
    OpenAI = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def default_experiments():
    """Return a dictionary mapping experiment names to message lists.

    Only five experiments are returned to keep runtime low while still
    exercising a variety of routing paths.
    """
    return {
        "1_factoid_trivial": [
            {"role": "user", "content": "What is the capital of Mongolia?"}
        ],
        "2_reasoning_deep": [
            {
                "role": "user",
                "content": (
                    "Carefully reason step by step: "
                    "If a train leaves Paris at 9:00am traveling 80 km/h, "
                    "and another leaves Berlin at 10:30am traveling 120 km/h toward Paris, "
                    "when and where will they meet? State assumptions and show your math."
                ),
            }
        ],
        "3_context_long": [
            {
                "role": "user",
                "content": (
                    "Summarize the following long text in 5 bullet points:\n\n"
                    + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 300)
                    + "\n\nPlease extract only the most salient points."
                ),
            }
        ],
        "4_creative_sonnet": [
            {
                "role": "user",
                "content": "Write a 12-line Shakespearean sonnet about black holes and quantum physics.",
            }
        ],
        "5_code_write": [
            {
                "role": "user",
                "content": (
                    "Write a Python function that implements the Sieve of Eratosthenes "
                    "to generate all primes up to 10000."
                ),
            }
        ],
    }


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 5.0) -> None:
    """Sleep with exponential backoff."""
    delay = min(cap, base * (2**attempt))
    time.sleep(delay)


def ensure_sdk() -> None:
    if not HAS_SDK:
        raise RuntimeError(
            "OpenAI SDK not found. Install with:\n  uv pip install openai python-dotenv"
        )


# ---------------------------------------------------------------------------
# Core probing logic
# ---------------------------------------------------------------------------

def run_probe(
    model: str,
    runs_per_experiment: int,
    seed: int | None,
    temperature: float,
    max_retries: int,
    sleep_between: float,
    out_dir: str,
    threads: int,
) -> None:
    """Execute the experiments against the GPT-5 router."""
    ensure_sdk()

    experiments = default_experiments()
    os.makedirs(out_dir, exist_ok=True)

    combined_rows = []
    experiment_to_counts: dict[str, Counter] = defaultdict(Counter)

    for exp_name, messages in experiments.items():
        rows: list[list[object]] = []

        def worker(run_index: int) -> tuple[list[object], str]:
            attempt = 0
            while True:
                start = time.time()
                try:
                    client = OpenAI(timeout=30)
                    kwargs = dict(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                    if seed is not None:
                        kwargs["seed"] = seed

                    resp = client.chat.completions.create(**kwargs)
                    end = time.time()
                    latency_ms = int((end - start) * 1000)

                    fp = getattr(resp, "system_fingerprint", None)
                    response_id = getattr(resp, "id", None)
                    usage = getattr(resp, "usage", None) or {}
                    prompt_tokens = getattr(usage, "prompt_tokens", None) or usage.get(
                        "prompt_tokens"
                    )
                    completion_tokens = getattr(usage, "completion_tokens", None) or usage.get(
                        "completion_tokens"
                    )
                    total_tokens = getattr(usage, "total_tokens", None) or usage.get(
                        "total_tokens"
                    )

                    row = [
                        now_iso(),
                        exp_name,
                        run_index + 1,
                        model,
                        fp,
                        response_id,
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    ]
                    if sleep_between > 0:
                        time.sleep(sleep_between)
                    return row, fp

                except Exception as e:  # pragma: no cover - network errors
                    if attempt >= max_retries:
                        end = time.time()
                        latency_ms = int((end - start) * 1000)
                        row = [
                            now_iso(),
                            exp_name,
                            run_index + 1,
                            model,
                            f"ERROR:{type(e).__name__}",
                            None,
                            latency_ms,
                            None,
                            None,
                            None,
                        ]
                        print(
                            f"[{exp_name} run {run_index+1}] ERROR after {attempt+1} attempts: {e}"
                        )
                        return row, f"ERROR:{type(e).__name__}"
                    attempt += 1
                    backoff_sleep(attempt)

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(worker, i) for i in range(runs_per_experiment)]
            for fut in tqdm(as_completed(futures), total=runs_per_experiment, desc=exp_name):
                row, fp = fut.result()
                rows.append(row)
                combined_rows.append(row)
                experiment_to_counts[exp_name][fp] += 1

        rows.sort(key=lambda r: r[2])  # sort by run_index
        csv_path = os.path.join(out_dir, f"{exp_name}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_utc",
                    "experiment",
                    "run_index",
                    "requested_model",
                    "system_fingerprint",
                    "response_id",
                    "latency_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ]
            )
            writer.writerows(rows)

    combined_csv = os.path.join(out_dir, "combined_results.csv")
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_utc",
                "experiment",
                "run_index",
                "requested_model",
                "system_fingerprint",
                "response_id",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ]
        )
        writer.writerows(combined_rows)

    summary = {exp: dict(counter) for exp, counter in experiment_to_counts.items()}
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Router Fingerprint Summary (by experiment) ===")
    for exp, counter in experiment_to_counts.items():
        print(f"\n{exp}:")
        total = sum(counter.values())
        for fp, c in counter.most_common():
            pct = (c / total * 100) if total else 0
            print(f"  {fp}: {c} / {total} ({pct:.1f}%)")

    print(
        f"\nWrote per-experiment CSVs, combined_results.csv, and summary.json to: {out_dir}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe GPT-5 router variability across experiments."
    )
    parser.add_argument("--model", default="gpt-5", help="Model to call (routered alias).")
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of runs per experiment.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for sampling (optional).")
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries on transient errors."
    )
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=0.2,
        help="Sleep seconds between calls.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads to use for concurrent calls.",
    )
    args = parser.parse_args()

    if os.environ.get("OPENAI_API_KEY") in (None, "", "YOUR_API_KEY"):
        print(
            "WARNING: OPENAI_API_KEY not set. Export it before running:\n  export OPENAI_API_KEY=sk-..."
        )
    run_probe(
        model=args.model,
        runs_per_experiment=args.runs,
        seed=args.seed,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_between=args.sleep_between,
        out_dir=args.out_dir,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
