import os
import sys
from pathlib import Path

# judge.py reads OPENAI_API_KEY at EvalRunner construction time; tests never make a
# network call, so a placeholder is enough.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
