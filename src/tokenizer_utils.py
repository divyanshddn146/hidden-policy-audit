from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SpanTokens:
    start: int
    end: int
    token_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end - self.start


def require_fast_tokenizer(tokenizer: Any) -> None:
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required because intervention spans use offset mappings. "
            "Load Qwen with AutoTokenizer(..., use_fast=True)."
        )


def all_substring_spans(text: str, substring: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index < 0:
            break
        spans.append((index, index + len(substring)))
        start = index + 1
    return spans


def char_span_to_token_span(
    offsets: Iterable[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    token_indices: list[int] = []
    for index, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if tok_start < char_end and tok_end > char_start:
            token_indices.append(index)
    if not token_indices:
        raise ValueError(
            f"No tokens overlap character span [{char_start}, {char_end})."
        )
    return min(token_indices), max(token_indices) + 1


def tokenize_named_span(tokenizer: Any, text: str, name: str, occurrence: int = 0) -> SpanTokens:
    require_fast_tokenizer(tokenizer)
    spans = all_substring_spans(text, name)
    if occurrence >= len(spans):
        raise ValueError(
            f"Could not find occurrence {occurrence} of {name!r} in text: {text!r}"
        )
    char_start, char_end = spans[occurrence]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_start, token_end = char_span_to_token_span(
        encoded["offset_mapping"], char_start, char_end
    )
    ids = tuple(int(value) for value in encoded["input_ids"][token_start:token_end])
    return SpanTokens(token_start, token_end, ids)


def render_chat(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def continuation_token_ids(tokenizer: Any, messages: list[dict[str, str]], answer: str) -> list[int]:
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Qwen chat-template prompt is not an exact prefix of the completed conversation."
        )
    continuation = list(full_ids[len(prompt_ids) :])
    if not continuation:
        raise RuntimeError(f"No continuation tokens found for answer {answer!r}")
    return continuation
