import json, re
nb = json.load(open("solution.ipynb")); cells = nb["cells"]
code = [c for c in cells if c["cell_type"] == "code"]
md = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
out = ""
for c in code:
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream": out += "".join(o["text"])
        elif "data" in o: out += "".join(o["data"].get("text/plain", ""))
print(len(cells), "cells;", len(code), "code; executed", sum(1 for c in code if c.get("execution_count")),
      "; errors", sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"),
      "; code cells with output", sum(1 for c in code if c.get("outputs")))
heads = [l for c in cells if c["cell_type"] == "markdown" for l in "".join(c["source"]).split("\n") if l.startswith("#")]
ids = [f"{a}.{b}" for a, n in [(1, 5), (2, 6), (3, 6), (4, 4), (5, 6)] for b in range(1, n + 1)]
print("task headings once:", sum(1 for t in ids if sum(f"(Task {t}" in h for h in heads) == 1), "/27")
print("stage blocks:", md.count("Objective and Tasks of this Section"), "/5 | criteria blocks:", md.count("**Evaluation criteria of this section"), "/5 | question quotes:", md.count("> **Brief Question"), "/10")
print("labels removed ('as given in the brief'):", md.count("as given in the brief") == 0)
for k in ["## Stages Overview", "## How You're Evaluated (100 Marks)", "## Process", "### Problem Statement", "### Data Dictionary"]:
    print(f"  {k!r}:", md.count(k))
# numbers quoted in the readings -> must appear in outputs
Q = ["Rows removed: 940", "Rows flagged as outliers: 1749", "Rows after the second rule: 172914", "Rows above it: 174", "99.9th percentile of total_items: 21.9",
 "36173", "11222", "1749", "172914.00", "21.00", "18920.00", "1703.35", "9.23", "Train: (138331, 24)", "Test : (34583, 24)",
 "3.678", "4.692", "0.744", "8.135", "1.763", "0.963", "5.134", "5.195", "4.573", "0.756",
 "121039 17292 34583", "46.09 9.22", "Trainable parameters: 11521", "Loss before training (scaled units): 1.167",
 "Best epoch -> 23 with validation RMSE 4.495", "validation RMSE 4.541", "4.332",
 "4.482", "4.513", "4.192", "4.566", "4.490", "4.497", "4.509", "4.540", "4.537", "4.528", "4.512", "4.503", "4.485", "4.500", "4.421",
 "Best epoch: 53 | validation RMSE: 4.472", "'hidden': (32,)", "19.735", "22.015", "4.442", "3.494", "0.770", "7.713",
 "51.51", "49.72", "74.67", "72.39", "97.62", "96.53", "12.68", "13.44", "0.250", "0.229", "0.271", "0.185", "0.168", "0.203", "100.0",
 "4.434", "4.454", "4.688", "4.697", "833", "1.373706e+07", "2.929", "0.014", "5.071", "4.859", "4.970", "4.756", "4.690", "4.405", "3.940", "3.629",
 "0.212", "0.285", "0.311", "4.691", "4.654", "3.665", "0.748", "Reloaded network gives identical predictions: True", "42 minutes", "0.1"]
missing = [q for q in Q if q not in out]
print("quoted numbers missing from outputs:", missing if missing else f"none ({len(Q)} checked)")
# stale-number scan in the markdown
stale = [s for s in ["34,618", "138,470", "13,000", "about 0.17 minutes", "five more sides", "five experiments"] if s in md]
print("stale phrases left in text:", stale if stale else "none")
