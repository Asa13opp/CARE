"""Pluggable backends for rubric generation.

Each backend exposes a single method:
    generate(messages_batch: list[list[dict]]) -> list[str]

where messages_batch[i] is an OpenAI-style chat messages list and the return
value is one raw completion string per element.

Select a backend with load_backend(name, **kwargs).
Supported names: "vllm", "openai".
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class RubricBackend(ABC):
    @abstractmethod
    def generate(self, messages_batch: list[list[dict]]) -> list[str]:
        """Return one raw completion string per messages list, in the same order."""


class VLLMBackend(RubricBackend):
    """Local model via vLLM (PagedAttention, continuous batching)."""

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        max_tokens: int = 768,
        temperature: float = 0.0,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        print(f"[VLLMBackend] Loading {model} (tp={tensor_parallel_size}) …", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

    def _apply_template(self, messages: list[dict]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def generate(self, messages_batch: list[list[dict]]) -> list[str]:
        prompts = [self._apply_template(msgs) for msgs in messages_batch]
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [o.outputs[0].text for o in outputs]


class OpenAIBackend(RubricBackend):
    """OpenAI-compatible API backend (remote or local server).

    Useful for GPT-4o, DeepSeek, Claude-via-proxy, or any llama.cpp / vLLM server.
    Reads credentials from env if not passed explicitly:
        RUBRIC_API_BASE_URL, RUBRIC_API_KEY, RUBRIC_API_MODEL
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 768,
        temperature: float = 0.0,
        max_workers: int = 16,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        import httpx
        from openai import OpenAI

        self._model = model or os.environ["RUBRIC_API_MODEL"]
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_workers = max_workers
        self._extra_body = extra_body or {}
        self._response_format = response_format
        self._reasoning_effort = reasoning_effort
        self._client = OpenAI(
            base_url=base_url or os.environ.get("RUBRIC_API_BASE_URL"),
            api_key=api_key or os.environ.get("RUBRIC_API_KEY", "EMPTY"),
            timeout=httpx.Timeout(float(os.environ.get("RUBRIC_API_TIMEOUT", "120")), connect=10.0),
            max_retries=5,
        )
        print(
            f"[OpenAIBackend] model={self._model} base_url={self._client.base_url} "
            f"workers={self._max_workers}",
            flush=True,
        )

    def _call_one(self, messages: list[dict]) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "extra_body": self._extra_body or None,
        }
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except TypeError as exc:
            if "reasoning_effort" not in kwargs and "response_format" not in kwargs:
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("reasoning_effort", None)
            fallback_kwargs.pop("response_format", None)
            try:
                resp = self._client.chat.completions.create(**fallback_kwargs)
            except TypeError:
                raise exc
        choice = resp.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            print(
                "[OpenAIBackend] rubric generation truncated "
                f"(model={self._model}, max_tokens={self._max_tokens})",
                flush=True,
            )
        return choice.message.content or ""

    def generate(self, messages_batch: list[list[dict]]) -> list[str]:
        results: list[str] = [""] * len(messages_batch)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._call_one, msgs): i for i, msgs in enumerate(messages_batch)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results


def load_backend(name: str, **kwargs: Any) -> RubricBackend:
    """Factory: load_backend("vllm", model=..., tensor_parallel_size=2)"""
    name = name.lower()
    if name == "vllm":
        return VLLMBackend(**kwargs)
    if name == "openai":
        return OpenAIBackend(**kwargs)
    raise ValueError(f"Unknown rubric backend: {name!r}. Choose from: vllm, openai")
