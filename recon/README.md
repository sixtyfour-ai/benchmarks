# RECON Benchmark

**RE**search & **CON**firmation — evaluating AI systems on verified people research.

100 real people, 474 verified fields, binary judging (correct / wrong / missing).

## Results

| Provider | Configuration | Accuracy | Weighted Accuracy | Precision | Latency P50 |
|----------|--------------|----------|-------------------|-----------|-------------|
| Sixtyfour | High | 83.5% | 73.0% | 88.8% | 328s |
| Sixtyfour | Medium | 67.9% | 61.0% | 90.7% | 310s |
| Parallel | Ultra 8x | 70.5% | 60.6% | 87.7% | 990s |
| Parallel | Ultra 2x | 64.1% | 56.8% | 89.8% | 559s |
| Sixtyfour | Low | 51.7% | 45.1% | 88.8% | 78s |
| xAI | Grok 4.20 Multi-Agent | 56.1% | 42.2% | 80.1% | 194s |
| Parallel | Ultra | 46.0% | 39.8% | 88.1% | 540s |
| OpenAI | GPT-5.4 xhigh | 36.8% | 32.9% | 90.5% | 224s |
| Google | Gemini 3.1 Pro | 32.6% | 22.8% | 76.9% | 82s |
| Exa | Search Deep Reasoning | 22.7% | 15.3% | 75.6% | 31s |

**Weighted accuracy** = (correct − wrong) / total_fields. Penalizes hallucination.

## Reproducing

### 1. Setup

```bash
cd benchmarks/recon
pip install httpx openai python-dotenv
```

### 2. Dataset

Place `people_data.json` in `data/`. This file contains 100 people with 474 verified fields. It is not included in the repo — request access from Sixtyfour or download from the provided S3 presigned URL.

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

Each script runs all 100 people by default. Use `--people N` for a smaller test.

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
python scripts/grok.py --model 4.20-ma           # Grok 4.20 Multi-Agent (RECON config)
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
| Grok 4.20 Multi-Agent | `grok.py` | `--model 4.20-ma` |
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
    "correct": 245,
    "wrong": 31,
    "missing": 198,
    "total_fields": 474,
    "accuracy": 51.7
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
