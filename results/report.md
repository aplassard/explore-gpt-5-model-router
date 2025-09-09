# GPT-5 Router Probe Results

This run executed one call for each of the five example prompts. The `system_fingerprint` field was `null` in every response, indicating no observable routing differences for any prompt.

| Experiment | System Fingerprint | Count |
|------------|--------------------|-------|
| 1_factoid_trivial | null | 1 |
| 2_reasoning_deep | null | 1 |
| 3_context_long | null | 1 |
| 4_creative_sonnet | null | 1 |
| 5_code_write | null | 1 |

No variability in the router fingerprint was detected within or across prompts. To increase confidence, rerun the probe with more samples:

```bash
uv run python router_probe.py --runs 100 --temperature 1 --out-dir results
```
