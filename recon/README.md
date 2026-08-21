# RECON Benchmark

**RE**search & **CON**firmation — evaluating AI systems on verified people research.

140 real people, 514 verified fields, binary judging (correct / wrong / missing).

## Results

| Provider | Configuration | Accuracy | Weighted Accuracy | Precision | Latency P50 |
|----------|--------------|----------|-------------------|-----------|-------------|
| Sixtyfour | High | 71.8% | +56.6% | 82.6% | 459s |
| Sixtyfour | Medium | 58.8% | +45.3% | 81.4% | 223s |
| Parallel | Ultra 8x | 44.6% | +31.3% | 77.1% | 678s |
| Sixtyfour | Low | 42.4% | +29.4% | 76.5% | 230s |
| Parallel | Ultra 2x | 42.6% | +28.8% | 75.5% | 834s |
| Parallel | Ultra | 44.4% | +23.9% | 68.5% | 589s |
| OpenAI | GPT-5.4 xhigh | 29.4% | +23.2% | 82.5% | 180s |
| Google | Gemini 3.1 Pro | 26.9% | +16.6% | 72.3% | 87s |
| xAI | Grok 4.3 | 24.5% | +10.1% | 63.0% | 15s |
| Exa | Search Deep | 12.7% | −5.4% | 41.3% | 4s |
| Exa | Search Deep Reasoning | 19.7% | −10.4% | 39.6% | 13s |

**Weighted accuracy** = (correct − wrong) / total_fields. Penalizes hallucination.

Two scoring views ship in this harness and they are not interchangeable:

| view | definition | where it appears |
|---|---|---|
| **pooled** (published) | `(correct − wrong) / total_fields` over every field | the table above, and `weighted_formula` in `results/*.json` |
| **macro** (25%/bucket) | equal-weight mean of the four per-bucket scores | `summary.scores.overall`, printed by `leaderboard.py` |

They diverge whenever the buckets differ in size. The published table is the pooled view.

**Judge health.** Every run summary carries a `judge_health` block:

```json
"judge_health": {"degraded_fields": 0, "degraded_pct": 0.0, "breakdown": {}, "trustworthy": true}
```

A field is *degraded* when the judge could not adjudicate it and the harness fell back to
substring containment (`judge_fields` retries `JUDGE_MAX_ATTEMPTS` times first). Containment
is a guess: it scores `"ranked 19th"` as correct against a ground truth of `"9th"`. Degraded
fields are tagged individually and counted here, so a score is never reported without the
figure that qualifies it. A run above `JUDGE_DEGRADED_WARN_PCT` is marked
`"trustworthy": false` and should not be published.

**Dataset pinning.** Set `PEOPLE_DATA_SHA256` to the `dataset_sha256` from the results file
you are reproducing and `load_people()` will refuse to run against a different dataset.

## Reproducing

### 1. Setup

```bash
cd benchmarks/recon
pip install httpx openai python-dotenv
```

### 2. Dataset

Place `people_data.json` in `data/`. This file contains 140 people with 514 verified fields. It is not included in the repo — request access from Sixtyfour or download from the provided S3 presigned URL.

### 3. API keys

Create a `.env` file in the repo root (`benchmarks/.env`):

```env
# Required by all scripts (judge uses GPT-4.1-mini)
OPENAI_API_KEY=your-key

# Per-provider keys — only needed for the scripts you run
SIXTYFOUR_API_KEY=your-key
GEMINI_API_KEY=your-key
XAI_API_KEY=your-key
EXA_API_KEY=your-key
PARALLEL_API_KEY=your-key
```

Get a Sixtyfour API key at [app.sixtyfour.ai/keys](https://app.sixtyfour.ai/keys).

### 4. Run a provider

Each script runs all 140 people by default. Use `--people N` for a smaller test.

```bash
# Sixtyfour (default: low tier)
python scripts/sixtyfour.py --tier low
python scripts/sixtyfour.py --tier medium
python scripts/sixtyfour.py --tier high          # requires access — contact sales

# OpenAI GPT
python scripts/gpt.py                            # default: gpt-5.4, reasoning=xhigh
python scripts/gpt.py --model gpt-5.4 --reasoning high

# Google Gemini
python scripts/gemini.py                          # default: gemini-3.1-pro-preview, thinking=high
python scripts/gemini.py --thinking medium

# xAI Grok
python scripts/grok.py --model 4.3               # Grok 4.3 (RECON config)
python scripts/grok.py --model 4.1-fast

# Exa
python scripts/exa.py                            # default: deep-reasoning
python scripts/exa.py --type deep

# Parallel
python scripts/parallel.py --processor ultra     # default
python scripts/parallel.py --processor ultra8x   # supports --resume for crash recovery
```

### 5. RECON-exact configurations

These are the exact configs used to produce the published RECON numbers:

| Provider | Script | Command |
|----------|--------|---------|
| Sixtyfour Low | `sixtyfour.py` | `--tier low` |
| Sixtyfour Medium | `sixtyfour.py` | `--tier medium` |
| Sixtyfour High | `sixtyfour.py` | `--tier high` |
| GPT-5.4 xhigh | `gpt.py` | `--model gpt-5.4 --reasoning xhigh` |
| Gemini 3.1 Pro | `gemini.py` | `--model gemini-3.1-pro-preview --thinking high` |
| Grok 4.3 | `grok.py` | `--model 4.3` |
| Exa Deep | `exa.py` | `--type deep` |
| Exa Deep Reasoning | `exa.py` | `--type deep-reasoning` |
| Parallel Ultra | `parallel.py` | `--processor ultra` |
| Parallel Ultra 2x | `parallel.py` | `--processor ultra2x` |
| Parallel Ultra 8x | `parallel.py` | `--processor ultra8x` |

### 6. Output

Results are saved to `results/runs/` as JSON with per-person verdicts:

```json
{
  "config": { ... },
  "summary": {
    "correct": 369,
    "wrong": 78,
    "missing": 67,
    "total_fields": 514,
    "accuracy": 71.8
  },
  "results": [
    {
      "person": "...",
      "correct": 3, "wrong": 0, "missing": 2,
      "verdicts": {
        "field_name": { "match": "correct", "reason": "..." }
      },
      "output": { "field_name": "value returned by provider" }
    }
  ]
}
```

## Judging

All providers are judged by the same GPT-4.1-mini judge with binary verdicts:

- **CORRECT**: Core factual information matches. Format differences are ignored ("$10M" vs "10 million").
- **WRONG**: Factually incorrect or irrelevant answer.
- **MISSING**: Provider returned empty/null for the field.

The judge has 98.5% agreement with human evaluators on a 200-field sample.

## Scoring

- **Accuracy** = correct / total_fields
- **Weighted Accuracy** = (correct − wrong) / total_fields
- **Precision** = correct / (correct + wrong)
