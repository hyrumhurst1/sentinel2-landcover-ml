"""
fill_paper.py — fill paper/paper.md {{tokens}} from data/results.json["paper_tokens"].
Run after run_study.py:  python src/fill_paper.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = os.path.join(ROOT, "data", "results.json")
paper = os.path.join(ROOT, "paper", "paper.md")
out = os.path.join(ROOT, "paper", "paper_filled.md")

if not os.path.exists(res):
    raise SystemExit("data/results.json not found. Run src/run_study.py first.")

tokens = json.load(open(res)).get("paper_tokens", {})
text = open(paper, encoding="utf-8").read()
missing = set()


def repl(m):
    k = m.group(1)
    if k in tokens and tokens[k] is not None:
        return str(tokens[k])
    missing.add(k)
    return m.group(0)


open(out, "w", encoding="utf-8").write(re.sub(r"\{\{(\w+)\}\}", repl, text))
print("Wrote", out)
print("Unfilled (excluding the {{like_this}} example):",
      sorted(missing - {"like_this"}) or "NONE")
