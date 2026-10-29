"""Greedy completion sanity check against a running OpenAI-compatible server, for A/B of
switches that could change numerics: each prompt twice at temperature 0, writes the texts and,
given REF.json, prints the common prefix length with the reference.  A bit-exact change must
reproduce the reference in full; a non-bit-exact one usually diverges somewhere (judge by accuracy).
usage: [URL=http://127.0.0.1:30000] [MODEL=served-name] python3 greedy_chk.py OUT.json [REF.json]"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("URL", "http://127.0.0.1:30000")
MODEL = os.environ.get("MODEL", "default")
PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Let's think step by step.",
    "Write a Python function that returns the n-th Fibonacci number using memoization.\n",
    "请用三句话介绍一下长江。\n",
]


def complete(p):
    req = urllib.request.Request(
        URL + "/v1/completions",
        data=json.dumps({"model": MODEL, "prompt": p, "max_tokens": 256, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["text"]


out = [[complete(p) for _ in range(2)] for p in PROMPTS]
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
for i, (a, b) in enumerate(out):
    print(f"prompt {i}: {len(a)} chars, repeat identical: {a == b}")
if len(sys.argv) > 2:
    ref = json.load(open(sys.argv[2]))
    for i, (a, r) in enumerate(zip(out, ref)):
        n = next((j for j, (x, y) in enumerate(zip(a[0], r[0])) if x != y), min(len(a[0]), len(r[0])))
        print(f"prompt {i}: common prefix with ref {n} / {len(r[0])} chars")
