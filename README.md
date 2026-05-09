# Sixtyfour Benchmarks

Open benchmarks for evaluating people and company research capabilities.

## Benchmarks

[RECON](recon/)

## Quick Start

```bash
git clone https://github.com/sixtyfour-ai/benchmarks.git
cd benchmarks/recon
pip install httpx openai python-dotenv
```

### Running a provider

```bash
export OPENAI_API_KEY="your-key"
python scripts/gpt.py

export GEMINI_API_KEY="your-key"
python scripts/gemini.py

export XAI_API_KEY="your-key"
python scripts/grok.py

export EXA_API_KEY="your-key"
python scripts/exa.py

export PARALLEL_API_KEY="your-key"
python scripts/parallel.py

export SIXTYFOUR_API_KEY="your-key"
python scripts/sixtyfour.py
```

## Requirements

- Python 3.11+
- OpenAI API key (for LLM judging)
- Provider API credentials for whichever runner you want to use

## License

MIT
