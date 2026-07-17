"""
Evaluate Exa search API on the RECON benchmark.

Two modes (Exa's two research products):
  agent   — POST /agent/runs (agentic research): full field set as one outputSchema plus an
            effort level, then poll GET /agent/runs/{id}; structured result in output.structured.
  search  — POST /search with type="deep-reasoning"/"deep" and an outputSchema, capped at
            10 properties per call, so fields are chunked and merged; output in output.content.

Usage:
    python scripts/exa.py --mode agent --effort xhigh
    python scripts/exa.py --mode search --type deep-reasoning
    python scripts/exa.py --mode agent --people 5
Requires: EXA_API_KEY, OPENAI_API_KEY in env
"""

import argparse
import asyncio
import json
import os
import time

import httpx
from judge import load_people, EvalRunner, clean_struct, post_with_retry

EXA_API_KEY = os.environ["EXA_API_KEY"]
BASE = "https://api.exa.ai"
SEARCH_MAX_PROPS = 10


def field_prompt(item: dict, fields: list[dict]) -> str:
    desc = "\n".join(f"- {f['fieldname']}: {f['description']}" for f in fields)
    return (
        f"You are a research agent. Given a description of a person, find specific facts about them.\n\n"
        f"Person: {item['person_info']}\n\nFind the following fields:\n{desc}\n\n"
        f"Search thoroughly; if you cannot find a definitive answer, return an empty string for that field."
    )


def schema_for(fields: list[dict]) -> dict:
    props = {f["fieldname"]: {"type": "string", "description": f["description"]} for f in fields}
    return {"type": "object", "properties": props, "required": list(props)}


async def call_search(client: httpx.AsyncClient, item: dict, search_type: str) -> dict:
    """One pass via /search: chunk fields into <=10-property schemas, merge structured outputs."""
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    merged, total_cost = {}, 0.0
    fields = item["fields"]
    for i in range(0, len(fields), SEARCH_MAX_PROPS):
        chunk = fields[i:i + SEARCH_MAX_PROPS]
        resp = await post_with_retry(client, f"{BASE}/search", headers=headers,
            json={"query": field_prompt(item, chunk), "type": search_type, "numResults": 10,
                  "contents": {"text": True}, "outputSchema": schema_for(chunk)})
        response = resp.json()
        out = response.get("output", {})
        if isinstance(out, dict) and "content" in out:
            out = out["content"]
        if isinstance(out, str):
            try:
                out = json.loads(out)
            except json.JSONDecodeError:
                out = {}
        if isinstance(out, dict):
            merged.update(out)
        cost = response.get("costDollars", {})
        total_cost += cost.get("total", 0) if isinstance(cost, dict) else (cost or 0)
    return {"output": merged, "cost": total_cost}


async def call_agent(client: httpx.AsyncClient, item: dict, effort: str) -> dict:
    """One pass via /agent/runs: submit the full field set, poll until completed."""
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    run_id = None
    for attempt in range(10):
        try:
            r = await client.post(f"{BASE}/agent/runs", headers=headers,
                json={"query": field_prompt(item, item["fields"]), "effort": effort,
                      "outputSchema": schema_for(item["fields"])})
        except (httpx.TransportError, httpx.TimeoutException, OSError):
            await asyncio.sleep(min(2 ** attempt, 45)); continue
        if r.status_code == 429 or r.status_code >= 500:
            await asyncio.sleep(min(2 ** attempt, 45)); continue
        r.raise_for_status()
        run_id = r.json()["id"]
        break
    if not run_id:
        raise RuntimeError("exa agent submit failed after retries")
    deadline = time.time() + 3600
    while time.time() < deadline:
        await asyncio.sleep(12)
        try:
            s = await client.get(f"{BASE}/agent/runs/{run_id}", headers=headers)
        except (httpx.TransportError, httpx.TimeoutException, OSError):
            await asyncio.sleep(6); continue
        if s.status_code == 429 or s.status_code >= 500:
            await asyncio.sleep(6); continue
        s.raise_for_status()
        data = s.json()
        st = data.get("status", "")
        if st in ("completed", "failed", "canceled", "cancelled"):
            if st != "completed":
                raise RuntimeError(f"exa agent {st}: {data.get('stopReason')}")
            out = (data.get("output") or {}).get("structured") or {}
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    out = {}
            cost = data.get("costDollars", {})
            return {"output": out, "cost": cost.get("total", 0) if isinstance(cost, dict) else (cost or 0)}
    raise TimeoutError("exa agent did not complete")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["agent", "search"], default="agent")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "auto"], default="xhigh",
                   help="agent mode only")
    p.add_argument("--type", choices=["deep", "deep-reasoning"], default="deep-reasoning",
                   help="search mode only")
    p.add_argument("--people", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=3)
    args = p.parse_args()

    people = load_people(args.people)
    name = f"exaagent_{args.effort}" if args.mode == "agent" else f"exa_{args.type}"
    runner = EvalRunner(name, vars(args))
    sem = asyncio.Semaphore(args.concurrency)
    print(f"Running Exa {args.mode} ({args.effort if args.mode == 'agent' else args.type}) "
          f"on {len(people)} people, concurrency={args.concurrency}", flush=True)

    async def process(item):
        async with sem:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    if args.mode == "agent":
                        res = await call_agent(client, item, args.effort)
                    else:
                        res = await call_search(client, item, args.type)
                output = clean_struct(res["output"], item["fields"])
                await runner.record(item, output, time.time() - t0, {"cost_usd": res["cost"]})
            except Exception as e:
                await runner.record_error(item, time.time() - t0, e)

    await asyncio.gather(*(process(p) for p in people))
    runner.summary()


if __name__ == "__main__":
    asyncio.run(main())
