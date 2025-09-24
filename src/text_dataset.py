from collections import Counter
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


def load_text_dataset(name: str) -> tuple[dict, str, str]:
    if name == "sst2":
        ds = load_dataset("glue", "sst2", cache_dir="~/pytorch_datasets")
        text_key = "sentence"
        label_key = "label"
        ds = {"train": ds["train"], "test": ds["validation"]}
    elif name == "trec":
        ds = load_dataset("trec", cache_dir="~/pytorch_datasets")
        text_key = "text"
        label_key = "coarse_label"
    elif name == "agnews":
        ds = load_dataset("ag_news", cache_dir="~/pytorch_datasets")
        text_key = "text"
        label_key = "label"
    else:
        msg = f"Dataset {name} is not supported."
        raise NotImplementedError(msg)

    return ds, text_key, label_key


@dataclass
class VocabInfo:
    pad_id: int
    vocab_size: int


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    train_size: int | None = None,
    max_len: int = 128,
) -> tuple[DataLoader, DataLoader, VocabInfo]:
    """Return DataLoader for training and testing, and the vocabulary information."""
    ds, text_key, label_key = load_text_dataset(dataset_name)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    counter = Counter()
    for ex in ds["train"]:
        tokens = tokenizer.encode(ex[text_key])
        counter.update(tokens)
    vocab = set(
        counter.keys()
        | {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        },
    )

    old2new = {old: new for new, old in enumerate(sorted(vocab))}
    pad_id = old2new[tokenizer.pad_token_id]
    vocab_size = len(vocab)

    def encode_fn(text: str) -> dict:
        ids = tokenizer.encode(text)[:max_len]
        return [
            old2new.get(token_id, old2new[tokenizer.unk_token_id]) for token_id in ids
        ]

    def _encode(examples: dict) -> dict:
        ids = encode_fn(examples[text_key])
        return {"input_ids": ids, "length": len(ids)}

    train_ds = ds["train"].map(
        _encode,
        remove_columns=[c for c in ds["train"].column_names if c != label_key],
    )
    test_ds = ds["test"].map(
        _encode,
        remove_columns=[c for c in ds["test"].column_names if c != label_key],
    )

    if train_size is not None:
        train_ds = train_ds.select(range(train_size))

    def collate_fn(batch: list) -> tuple:
        input_ids = [ex["input_ids"] for ex in batch]
        lengths = [ex["length"] for ex in batch]
        labels = [ex[label_key] for ex in batch]

        max_batch_len = max(lengths)
        padded_input_ids = [
            ids + [pad_id] * (max_batch_len - len(ids))
            if len(ids) < max_batch_len
            else ids
            for ids in input_ids
        ]

        return (
            torch.tensor(padded_input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader, VocabInfo(pad_id=pad_id, vocab_size=vocab_size)
