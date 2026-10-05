import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_PATH, apply_chat_template, make_messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="explicit SFT checkpoint directory")
    parser.add_argument("--base-model", default=MODEL_PATH)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", default=str(project_root / "data" / "val.jsonl"))
    parser.add_argument("--output", default=str(project_root / "output" / "eval" / "sft" / "predictions.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tokenizer_path = args.model if Path(args.model).exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data, "r", encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for idx, line in enumerate(f):
            if args.limit and idx >= args.limit:
                break
            row = json.loads(line)
            prompt = apply_chat_template(tokenizer, make_messages(row["context"]), add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            pred_ids = generated[0, inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
            out.write(json.dumps({"id": row["id"], "dialog_id": row["dialog_id"], "prediction": pred, "target": row["target"]}, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
