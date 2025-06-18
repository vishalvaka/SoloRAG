#!/usr/bin/env python
"""scripts/evaluate.py
Quick smoke-test for the RAG pipeline.

Reads a tiny dev-set JSONL (question + expected keywords), sends each
question through *exactly the same* retrieval → prompt → Ollama path
that powers the FastAPI endpoint, then prints:

• per-query latency + pass/fail (all keywords found?)
• aggregate hit-rate & latency stats

Intended for local regression checks or lightweight CI gating.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import statistics
import time
from typing import List
import sys
import textwrap

# ensure project root is on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import retrieval  # reuse the live pipeline

DEV_SET = pathlib.Path("data/eval/dev_set.jsonl")


async def _one_pass(question: str, keywords: List[str]):
    start = time.perf_counter()
    answer, ctx = await retrieval.get_answer(question)
    latency = time.perf_counter() - start
    hit = all(k.lower() in answer.lower() for k in keywords)
    return latency, hit, answer, ctx


async def main():
    assert DEV_SET.exists(), (
        "Create a tiny dev-set first → data/eval/dev_set.jsonl (see README)."
    )

    latencies: list[float] = []
    hits: list[bool] = []

    for line in DEV_SET.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        lat, hit, answer, ctx = await _one_pass(row["question"], row["keywords"])
        latencies.append(lat)
        hits.append(hit)
        status = "✅" if hit else "❌"
        print(f"{status} {row['question'][:42]:<42} {lat*1000:6.1f} ms")
        if not hit:
            print("   → Expected keywords:", ", ".join(row["keywords"]))
            print("   → Answer snippet   :", answer[:160].replace("\n", " ") + ("…" if len(answer) > 160 else ""))
            print("   → Top context txt  :", textwrap.shorten(ctx[0]['text'], width=160, placeholder='…'))

    print("\n── Summary ─────────")
    print(f"Queries          : {len(latencies)}")
    print(f"Hit-rate         : {sum(hits)}/{len(hits)}  "
          f"({(sum(hits)/len(hits))*100:.0f}%)")
    print(f"Avg latency      : {statistics.mean(latencies):.2f} s")
    print(f"p95 latency      : {statistics.quantiles(latencies, n=20)[18]:.2f} s")


if __name__ == "__main__":
    asyncio.run(main())
