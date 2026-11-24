"""Copy outputs from an executed run into a freshly built notebook (same code cells, different markdown)."""
import json, sys
base_p, exec_p, out_p = sys.argv[1:4]
base, ex = json.load(open(base_p)), json.load(open(exec_p))
bc = [c for c in base["cells"] if c["cell_type"] == "code"]
ec = [c for c in ex["cells"] if c["cell_type"] == "code"]
assert len(bc) == len(ec), (len(bc), len(ec))
for b, e in zip(bc, ec):
    assert "".join(b["source"]) == "".join(e["source"]), "code differs:\n" + "".join(b["source"])[:200]
    b["outputs"], b["execution_count"] = e["outputs"], e["execution_count"]
json.dump(base, open(out_p, "w"), indent=1)
print("merged", len(bc), "code cells ->", out_p)
