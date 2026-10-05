from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from torch.utils.data import Dataset

SYSTEM_PROMPT = (
    "You are a helpful, warm, and empathetic AI assistant. "
    "Respond to the user's latest message in a supportive and natural way."
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class RubricResponseDataset(Dataset):
    def __init__(self, data_path: str, num_samples: int | None = None):
        data = load_jsonl(data_path)
        self.data = [item for item in data if item.get("rubric_valid", True) and item.get("rubric")]
        if num_samples and num_samples < len(self.data):
            self.data = random.sample(self.data, num_samples)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.data[idx])
        # history already ends with the current user query
        history = item.get("history") or []
        prompt = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        item["prompt"] = prompt
        return item
