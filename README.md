# distilbert-finetune

Small HuggingFace fine-tuning project (DistilBERT on IMDB).

## Structure
See `src/` for scripts: `train.py`, `data_utils.py`, `models.py`, `eval.py`.

## Setup
1. Create env:
   ```bash
   python -m venv venv && source venv/bin/activate
   ```

2. Install:

   ```bash
   git clone [https://github.com/AbdelkaderHamdi/distilbert-finetune.git](https://github.com/AbdelkaderHamdi/distilbert-finetune.git)
   cd distilbert-finetune
   pip install -r requirements.txt
   ```

## Quick train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train_subset 1000 --val_subset 500 --test_subset 500 \
  --output_dir models/distilbert-imdb
```

## Eval

```bash
python src/eval.py --model_dir models/distilbert-imdb --test_data data/processed/test.jsonl
```
