"""Print the text outputs of an executed notebook, with the nearest heading, so readings can be written from real numbers."""
import json, sys
nb = json.load(open(sys.argv[1]))
head, k = "", 0
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        for l in src.split("\n"):
            if l.startswith("#"): head = l
        continue
    k += 1
    outs = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream": outs.append("".join(o["text"]))
        elif o.get("output_type") in ("execute_result", "display_data"):
            outs.append("".join(o["data"].get("text/plain", "")) if "image/png" not in o["data"] else "[image]")
        elif o.get("output_type") == "error": outs.append("ERROR: " + o["ename"] + " " + o["evalue"])
    txt = "\n".join(outs)
    if txt.strip() and txt.strip() != "[image]":
        print(f"=== cell {i} (code #{k}) | {head[:70]}\n{src[:160].splitlines()[0] if src else ''}\n{txt[:1800]}\n")
