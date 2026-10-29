"""Summarise evalscope aime26 runs (repeats > 1): overall / per-repeat accuracy, accuracy of
answers cut at max_tokens vs finished ones, and per-problem correct counts side by side.
usage: python3 ana_aime.py <eval_dir_of_tag> [...]   e.g. ana_aime.py eval/opt eval/r1"""
import glob
import json
import sys

MAXTOK = 65536
runs = {}
for d in sys.argv[1:]:
    tag = d.rstrip("/").split("/")[-1]
    out = {}
    for l in open(glob.glob(f"{d}/aime26/*/predictions/*/*.jsonl")[0]):
        p = json.loads(l)
        out[p["index"]] = p["model_output"]["usage"]["output_tokens"]
    probs = {}  # question text -> [(index, acc, tokens)] in index order
    for l in open(glob.glob(f"{d}/aime26/*/reviews/*/*.jsonl")[0]):
        r = json.loads(l)
        q = r["messages"][0]["content"] if r["messages"] else str(r["target"])
        probs.setdefault(q, []).append((r["index"], r["sample_score"]["score"]["value"]["acc"], out[r["index"]]))
    for v in probs.values():
        v.sort()
    runs[tag] = probs
    allv = [x for v in probs.values() for x in v]
    nrep = max(len(v) for v in probs.values())
    rep = [sum(v[k][1] for v in probs.values() if len(v) > k) / len(probs) for k in range(nrep)]
    cut = [x for x in allv if x[2] >= MAXTOK]
    fin = [x for x in allv if x[2] < MAXTOK]
    tok = sorted(x[2] for x in allv)
    print(f"== {tag}: {len(probs)} problems x {nrep}, acc {sum(x[1] for x in allv) / len(allv):.4f}")
    print("   per repeat: " + " ".join(f"{a:.4f}" for a in rep))
    print(f"   finished {len(fin)}: acc {sum(x[1] for x in fin) / max(len(fin), 1):.4f};"
          f" cut at {MAXTOK}: {len(cut)}, acc {sum(x[1] for x in cut) / max(len(cut), 1):.4f}")
    print(f"   output tokens median {tok[len(tok) // 2]}, mean {sum(tok) // len(tok)}")
    print(f"   problems solved >=1/{nrep}: {sum(any(x[1] for x in v) for v in probs.values())},"
          f" {nrep}/{nrep}: {sum(all(x[1] for x in v) for v in probs.values())}")

if len(runs) > 1:
    tags = list(runs)
    qs = list(runs[tags[0]])
    print("\nper problem (# correct of repeats): " + " / ".join(tags))
    diff = 0
    for i, q in enumerate(qs):
        c = [int(sum(x[1] for x in runs[t].get(q, []))) for t in tags]
        diff += c[0] != c[-1]
        print(f"  P{i + 1:02d} " + " / ".join(map(str, c)) + ("  *" if len(set(c)) > 1 else ""))
    print(f"problems with a different count: {diff}")
