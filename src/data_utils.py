from datasets import load_dataset
from transformers import DataCollatorWithPadding


def load_imdb_split(train_subset=1000, val_subset=500, test_subset=500, seed=42):
    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=0.2, seed=seed)
    train = split["train"]
    # split the 20% into val/test evenly so test remains unseen
    val, test = split["test"].train_test_split(test_size=0.5, seed=seed).values()

    # optional small-subset selection for quick experiments
    if train_subset is not None:
        train = train.shuffle(seed=seed).select(range(min(len(train), train_subset)))
    if val_subset is not None:
        val = val.shuffle(seed=seed).select(range(min(len(val), val_subset)))
    if test_subset is not None:
        test = test.shuffle(seed=seed).select(range(min(len(test), test_subset)))

    return train, val, test

def tokenize_dataset(dataset, tokenizer, max_length=256):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    return dataset.map(_tok, batched=True, remove_columns=["text"])

def get_data_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer)
