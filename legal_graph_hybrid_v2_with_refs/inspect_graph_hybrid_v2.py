
import argparse, json
from collections import Counter
from legal_graph_hybrid_v2.common import read_jsonl
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--graph-dir", required=True); ap.add_argument("--top-k", type=int, default=30); args=ap.parse_args()
    nodes=list(read_jsonl(f"{args.graph_dir}/nodes.jsonl")); edges=list(read_jsonl(f"{args.graph_dir}/edges.jsonl"))
    byid={n["id"]:n for n in nodes}; deg=Counter()
    for e in edges:
        if e.get("type")=="HAS_ENTITY": deg[e["target"]]+=1
    top=[]
    for nid,d in deg.most_common(args.top_k):
        n=byid.get(nid,{})
        top.append({"node":nid,"degree":d,"label":n.get("label"),"canonical":n.get("canonical")})
    ref_out=Counter(e["source"] for e in edges if e.get("type")=="REFERENCES")
    print(json.dumps({
        "node_types":dict(Counter(n.get("type") for n in nodes)),
        "edge_types":dict(Counter(e.get("type") for e in edges)),
        "top_entity_degree":top,
        "top_reference_out_degree":ref_out.most_common(args.top_k),
    },ensure_ascii=False,indent=2))
if __name__=="__main__": main()
