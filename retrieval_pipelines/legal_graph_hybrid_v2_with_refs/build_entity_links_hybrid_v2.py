
import argparse
from legal_graph_hybrid_v2.entity_links import build_hybrid_entity_links
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gazetteer-entities-root", required=True)
    ap.add_argument("--gliner-entities-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-inherited", action="store_true")
    args=ap.parse_args()
    print(build_hybrid_entity_links(args.gazetteer_entities_root,args.gliner_entities_root,args.output,not args.no_inherited))
if __name__=="__main__": main()
