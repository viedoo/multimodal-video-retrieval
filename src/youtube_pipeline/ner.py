from __future__ import annotations

import re
from typing import Any, Iterable


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOCATION",
    "LOCATION": "LOCATION",
    "MISC": "MISC",
    "MISCELLANEOUS": "MISC",
}


def text_chunks(text: str, max_chars: int = 600) -> Iterable[str]:
    current: list[str] = []
    size = 0
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        if current and size + len(sentence) + 1 > max_chars:
            yield " ".join(current)
            current = []
            size = 0
        if len(sentence) > max_chars:
            for offset in range(0, len(sentence), max_chars):
                yield sentence[offset : offset + max_chars]
            continue
        current.append(sentence)
        size += len(sentence) + 1
    if current:
        yield " ".join(current)


class ElectraVietnameseNER:
    def __init__(
        self,
        model: str = "NlpHUST/ner-vietnamese-electra-base",
        device: str = "auto",
        min_score: float = 0.55,
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "ELECTRA NER requires the optional dependency: "
                'pip install -e ".[ner]"'
            ) from exc

        if device == "auto":
            pipeline_device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("NER device=cuda requested but PyTorch cannot access CUDA.")
            pipeline_device = 0
        else:
            pipeline_device = -1
        self.min_score = min_score
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.model_max_length = 512
        self.pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=pipeline_device,
        )

    def normalize_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in items:
            score = float(item.get("score") or 0.0)
            if score < self.min_score:
                continue
            raw_label = str(item.get("entity_group") or item.get("entity") or "MISC")
            label = raw_label.removeprefix("B-").removeprefix("I-").upper()
            entity_type = LABEL_MAP.get(label, "MISC")
            value = str(item.get("word") or "").replace("_", " ").strip()
            value = value.replace(" ##", "").replace("##", "")
            key = (value.casefold(), entity_type)
            if len(value) < 2 or key in seen:
                continue
            seen.add(key)
            entities.append({"text": value, "type": entity_type, "score": score, "source": "electra"})
        return entities

    def extract(self, text: str) -> list[dict[str, Any]]:
        return self.normalize_items(
            [item for chunk in text_chunks(text) for item in self.pipeline(chunk)]
        )

    def extract_many(self, texts: list[str], batch_size: int = 32) -> list[list[dict[str, Any]]]:
        chunks: list[str] = []
        owners: list[int] = []
        for owner, text in enumerate(texts):
            for chunk in text_chunks(text):
                chunks.append(chunk)
                owners.append(owner)
        grouped: list[list[dict[str, Any]]] = [[] for _ in texts]
        if not chunks:
            return grouped
        outputs = self.pipeline(chunks, batch_size=batch_size)
        for owner, items in zip(owners, outputs):
            grouped[owner].extend(items)
        return [self.normalize_items(items) for items in grouped]
