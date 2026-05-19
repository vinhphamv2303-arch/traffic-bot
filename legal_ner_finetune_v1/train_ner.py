import argparse
from pathlib import Path
from legal_ner_finetune_v1.config import TrainConfig
from legal_ner_finetune_v1.trainer import train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-file', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model', default='FacebookAI/xlm-roberta-large')
    ap.add_argument('--epochs', type=float, default=5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--eval-ratio', type=float, default=0.1)
    ap.add_argument('--negative-ratio', type=float, default=0.0)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--no-auto-clean', action='store_true')
    args = ap.parse_args()
    cfg = TrainConfig(
        train_file=Path(args.train_file), output_dir=Path(args.output_dir), model_name_or_path=args.model,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, max_length=args.max_length, eval_ratio=args.eval_ratio,
        negative_ratio=args.negative_ratio, fp16=args.fp16, bf16=args.bf16, auto_clean=not args.no_auto_clean,
    )
    m = train(cfg)
    print('Fine-tuning completed')
    print(f"Final model: {m['final_model_dir']}")
    print(f"Rows: {m['row_count']} | train: {m['train_count']} | dev: {m['dev_count']}")
    print(f"Entities: {m['entity_count']}")

if __name__ == '__main__':
    main()
