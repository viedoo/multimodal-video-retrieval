from __future__ import annotations

from typing import Any


class TransformerReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError('Reranking requires: pip install -e ".[ner,retrieval]"') from exc

        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected = device
        if selected == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Reranker device=cuda requested but PyTorch cannot access CUDA.")

        self.torch = torch
        self.device = torch.device(selected)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()

    def score(self, query: str, passages: list[str], batch_size: int = 8) -> list[float]:
        scores: list[float] = []
        with self.torch.inference_mode():
            for offset in range(0, len(passages), batch_size):
                batch = passages[offset : offset + batch_size]
                pairs = [[query, passage] for passage in batch]
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        return scores
