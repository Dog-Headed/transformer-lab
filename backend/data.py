"""Character/word tokens and disjoint chronological train/validation windows."""
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from .config import MAX_TEXT_CHARACTERS

DATA_PATH = Path(__file__).parent / "data" / "tiny_shakespeare.txt"
DATASETS = {"shakespeare": DATA_PATH, "stories": DATA_PATH.with_name("tiny_stories.txt")}
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]+|[^\s]")


def read_corpus(dataset):
    if dataset != "stories_identity":
        return DATASETS[dataset].read_text(encoding="utf-8")
    # Explicit training data only: inference never looks up these answers.
    stories = DATASETS["stories"].read_text(encoding="utf-8").strip().split("\n\n")
    examples = json.loads(DATA_PATH.with_name("self_introduction.json").read_text(encoding="utf-8"))
    mixed = []
    for index, story in enumerate(stories):
        example = examples[index % len(examples)]
        mixed.extend([story, f"User: {example['question']}\nAssistant: {example['answer']}"])
    return "\n\n".join(mixed) + "\n"


def token_spans(text, mode):
    if mode == "character":
        return [(char, i) for i, char in enumerate(text)]
    spans = []
    for match in WORD_PATTERN.finditer(text):
        value = match.group()
        if re.fullmatch(r"[\u3400-\u9fff]+", value):
            import jieba
            jieba.setLogLevel(logging.WARNING)
            start = match.start()
            for word in jieba.cut(value, HMM=False):
                spans.append((word, start))
                start += len(word)
        else:
            spans.append((value, match.start()))
    return spans


def tokenize(text, mode):
    return [token for token, _ in token_spans(text, mode)]


def append_token(text, token, mode):
    if mode == "character" or not text:
        return text + token
    # Word models omit whitespace. English spaces are display formatting, not tokens.
    if re.match(r"[A-Za-z0-9]", token) and not text[-1].isspace() and text[-1] not in "(['\"“":
        return text + " " + token
    return text + token


class TokenDataset(Dataset):
    def __init__(self, tokens: list[str], vocabulary: list[str], sequence_length: int):
        lookup = {token: index for index, token in enumerate(vocabulary)}
        self.tokens = torch.tensor([lookup.get(token, 0) for token in tokens], dtype=torch.long)
        self.sequence_length = sequence_length
        # Non-overlapping inputs keep evaluation bounded; shifted labels remain next characters.
        self.windows = (len(self.tokens) - 1) // sequence_length
        if self.windows < 1:
            raise ValueError("训练集和验证集均须至少包含一个完整词元窗口，请增加语料或减小上下文")

    def __len__(self):
        return self.windows

    def __getitem__(self, index):
        start = index * self.sequence_length
        return (self.tokens[start:start + self.sequence_length],
                self.tokens[start + 1:start + self.sequence_length + 1])


def prepare_data(text: str, sequence_length: int, validation_ratio: float, mode="character"):
    if not 256 <= len(text) <= MAX_TEXT_CHARACTERS:
        raise ValueError(f"语料须包含 256–{MAX_TEXT_CHARACTERS} 个字符")
    spans = token_spans(text, mode)
    boundary = int(len(text) * (1 - validation_ratio))
    split = next((i for i, (_, start) in enumerate(spans) if start >= boundary), len(spans))
    if not split or split == len(spans):
        raise ValueError("语料不足以划分训练和验证词元")
    character_split = spans[split][1]
    training_tokens = [token for token, _ in spans[:split]]
    validation_tokens = [token for token, _ in spans[split:]]
    counts = Counter(training_tokens)
    if any(len(token) > 512 for token in counts):
        raise ValueError("单个词元不能超过 512 字符")
    limit = 4096 if mode == "word" else 256
    selected = sorted(counts, key=lambda token: (-counts[token], token))[:limit-1] if mode == "word" else counts
    vocabulary = ["<unk>"] + sorted(selected)
    if not 3 <= len(vocabulary) <= limit:
        raise ValueError(f"训练词表须有 2–{limit-1} 个不同词元")
    training = TokenDataset(training_tokens, vocabulary, sequence_length)
    validation = TokenDataset(validation_tokens, vocabulary, sequence_length)
    metadata = {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text), "train_characters": character_split,
        "validation_characters": len(text) - character_split, "vocabulary_size": len(vocabulary),
        "train_windows": len(training), "validation_windows": len(validation),
        "validation_unknown_characters": int((validation.tokens == 0).sum()) if mode == "character" else 0,
        "validation_unknown_tokens": int((validation.tokens == 0).sum()),
        "train_tokens": len(training_tokens), "validation_tokens": len(validation_tokens),
        "tokenizer": mode, "tokenizer_version": "word-regex-v1+jieba-0.42.1-HMM-off" if mode == "word" else "unicode-character-v1",
        "split": "chronological, disjoint character ranges; vocabulary from training only",
    }
    if mode == "word" and len(counts) >= limit:
        metadata.update(vocabulary_selection="4095 most frequent training tokens; lexical tie break; rare tokens mapped to <unk>",
                        train_unknown_tokens=int((training.tokens == 0).sum()),
                        distinct_training_tokens=len(counts))
    return training, validation, vocabulary, metadata
