\
import argparse
from legal_local_entity_extractor.train_gliner import train_gliner_model

def main():
    ap = argparse.ArgumentParser(description="Prepare/launch GLiNER fine-tuning from local NER dataset.")
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--dev-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="urchade/gliner_medium-v2.1")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    args = ap.parse_args()

    cfg = train_gliner_model(
        train_file=args.train_file,
        dev_file=args.dev_file,
        output_dir=args.output_dir,
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print("GLiNER training/preparation completed")
    print(f"Output: {args.output_dir}")
    if "warning" in cfg:
        print("WARNING:", cfg["warning"])

if __name__ == "__main__":
    main()
