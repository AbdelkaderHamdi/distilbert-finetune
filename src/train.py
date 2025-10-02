import argparse
import json
import os
import random
import numpy as np
import torch

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from src.data_utils import load_imdb_split, tokenize_dataset, get_data_collator
from src.models import load_tokenizer, build_model
from src.eval import calculate_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--output_dir", default="models/distilbert-imdb")
    p.add_argument("--train_subset", type=int, default=1000)
    p.add_argument("--val_subset", type=int, default=500)
    p.add_argument("--test_subset", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--freeze_base", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name)
    train_raw, val_raw, test_raw = load_imdb_split(
        train_subset=args.train_subset, val_subset=args.val_subset, test_subset=args.test_subset, seed=args.seed
    )

    train_tok = tokenize_dataset(train_raw, tokenizer)
    val_tok = tokenize_dataset(val_raw, tokenizer)
    test_tok = tokenize_dataset(test_raw, tokenizer)

    data_collator = get_data_collator(tokenizer)
    model = build_model(args.model_name, num_labels=2, freeze_base=args.freeze_base)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        eval_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=calculate_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset=test_tok)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training finished. Metrics saved to", os.path.join(args.output_dir, "metrics.json"))

if __name__ == "__main__":
    main()
