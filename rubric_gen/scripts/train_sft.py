import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from common import MODEL_PATH, SYSTEM_PROMPT, encode_completion_only


class RubricDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_seq_len: int, limit: int | None = None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                row = json.loads(line)
                messages = row.get("messages")
                if not isinstance(messages, list) or len(messages) != 3 or not all(
                    isinstance(message, dict) for message in messages
                ):
                    raise ValueError(f"{path}:{line_number}: expected exactly system/user/assistant messages")
                if messages[0].get("role") != "system" or messages[0].get("content") != SYSTEM_PROMPT:
                    raise ValueError(f"{path}:{line_number}: system prompt does not match the v5 schema")
                if messages[1].get("role") != "user" or messages[2].get("role") != "assistant":
                    raise ValueError(f"{path}:{line_number}: expected system/user/assistant message roles")
                self.rows.append(row)
                if limit and len(self.rows) >= limit:
                    break
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        encoded = encode_completion_only(self.tokenizer, self.rows[idx]["messages"], self.max_seq_len)
        return {k: encoded[k] for k in ("input_ids", "attention_mask", "labels")}


@dataclass
class DataCollator:
    tokenizer: any

    def __call__(self, features: List[Dict[str, List[int]]]):
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--train-file", default=str(project_root / "data" / "v5_accepted_build" / "train.jsonl"))
    parser.add_argument("--val-file", default=str(project_root / "data" / "v5_accepted_build" / "val.jsonl"))
    # Each run must write to its explicit run directory; no `latest` alias is maintained.
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--per-device-bs", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--tensorboard-dir", default=str(project_root / "logs" / "tensorboard"))
    parser.add_argument("--eval-strategy", default="epoch")
    parser.add_argument("--deepspeed", default=None,
                        help="path to a DeepSpeed config JSON (e.g. configs/ds_zero3.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_ds = RubricDataset(args.train_file, tokenizer, args.max_seq_len, args.train_limit)
    val_ds = RubricDataset(args.val_file, tokenizer, args.max_seq_len, args.val_limit)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = args.tensorboard_dir

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_bs,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_dir=args.tensorboard_dir,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy="epoch",
        save_total_limit=None,
        save_only_model=True,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        max_steps=args.max_steps,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        deepspeed=args.deepspeed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        # Trainer.save_model already writes the tokenizer through DataCollator. This file
        # only records training configuration and is not needed by from_pretrained().
        (Path(args.output_dir) / "training_args.bin").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
