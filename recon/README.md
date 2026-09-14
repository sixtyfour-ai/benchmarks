# RECON Benchmark

A benchmark for evaluating AI systems on verified people research.

140 real people, 514 verified fields, binary judging (correct / wrong / missing).

## Results

| Provider | Configuration | Accuracy | Weighted Accuracy | Precision | Correct / Wrong / Missing |
|----------|--------------|----------|-------------------|-----------|-------------|
| Sixtyfour | High | 67.7% | +54.3% | 83.5% | 348 / 69 / 97 |
| Sixtyfour | Medium | 56.0% | +44.7% | 83.2% | 288 / 58 / 168 |
| Parallel | Ultra 2x | 54.3% | +41.1% | 80.4% | 279 / 68 / 167 |
| Parallel | Ultra 8x | 52.1% | +37.5% | 78.1% | 268 / 75 / 171 |
| xAI | Grok 4.20-ma | 51.6% | +36.2% | 77.0% | 265 / 79 / 170 |
| xAI | Grok 4.6 | 50.8% | +36.0% | 77.4% | 261 / 76 / 177 |
| xAI | Grok 4.3 | 44.9% | +30.9% | 76.2% | 231 / 72 / 211 |
| Sixtyfour | Low | 49.8% | +28.8% | 70.3% | 256 / 108 / 150 |
| Parallel | Ultra | 43.4% | +27.2% | 72.9% | 223 / 83 / 208 |
| Exa | agent xhigh | 37.7% | +23.9% | 73.2% | 194 / 71 / 249 |
| OpenAI | GPT-5.6-sol xhigh | 31.3% | +20.4% | 74.2% | 161 / 56 / 297 |
| Google | Gemini 3.1 Pro (high) | 23.2% | +13.4% | 70.4% | 119 / 50 / 345 |
| DeepSeek | V4 Pro (high) | 12.6% | +9.3% | 79.3% | 65 / 17 / 432 |
| Anthropic | Claude Haiku 4.5 | 7.6% | +1.6% | 55.7% | 39 / 31 / 444 |

**Weighted accuracy** = (correct − wrong) / total_fields. Penalizes incorrect answers.

[Full results](results/sixtyfour_benchmark_results.json).

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
MOONSHOT_API_KEY=your-key
DEEPSEEK_API_KEY=your-key
ZAI_API_KEY=your-key
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
python scripts/gpt.py                            # default: gpt-5.6-sol, reasoning=xhigh
python scripts/gpt.py --model gpt-5.6-sol --reasoning high

# Native provider web-research harnesses
python scripts/kimi.py --reasoning max
python scripts/deepseek.py --reasoning none
python scripts/glm.py --reasoning max

# Google Gemini
python scripts/gemini.py                          # default: gemini-3.1-pro-preview, thinking=high
python scripts/gemini.py --thinking medium

# xAI Grok
python scripts/grok.py --model 4.3               # Grok 4.3 (RECON config)
python scripts/grok.py --model 4.1-fast

# Exa
python scripts/exa.py                            # default: agent, effort=xhigh
python scripts/exa.py --mode search --type deep

# Parallel
python scripts/parallel.py --processor ultra     # default
python scripts/parallel.py --processor ultra8x   # supports --resume for crash recovery
```

Kimi and GLM use bounded model-driven search loops; customize their budget with
`--max-search-rounds N`. DeepSeek executes and bounds web search server-side.

### 5. RECON configurations

Commands for configurations supported by these scripts:

| Provider | Script | Command |
|----------|--------|---------|
| Sixtyfour Low | `sixtyfour.py` | `--tier low` |
| Sixtyfour Medium | `sixtyfour.py` | `--tier medium` |
| Sixtyfour High | `sixtyfour.py` | `--tier high` |
| GPT-5.4 xhigh | `gpt.py` | `--model gpt-5.6-sol --reasoning xhigh` |
| Gemini 3.1 Pro | `gemini.py` | `--model gemini-3.1-pro-preview --thinking high` |
| Grok 4.3 | `grok.py` | `--model 4.3` |
| Exa agent xhigh | `exa.py` | `--mode agent --effort xhigh` |
| Parallel Ultra | `parallel.py` | `--processor ultra` |
| Parallel Ultra 2x | `parallel.py` | `--processor ultra2x` |
| Parallel Ultra 8x | `parallel.py` | `--processor ultra8x` |

### 6. Output

Results are saved to `results/runs/` as JSON with per-person verdicts:

```json
{
  "config": { ... },
  "summary": {
    "correct": 348,
    "wrong": 69,
    "missing": 97,
    "total_fields": 514,
    "accuracy": 67.7
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
