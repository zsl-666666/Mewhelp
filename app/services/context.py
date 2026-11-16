from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

TokenCounter = Callable[[list[BaseMessage]], int]


@dataclass(frozen=True)
class TrimmedHistory:
    messages: list[BaseMessage]
    trimmed: bool


def trim_history(
    messages: list[BaseMessage],
    max_tokens: int,
    token_counter: TokenCounter = count_tokens_approximately,
) -> TrimmedHistory:
    if not messages:
        raise ValueError("messages must not be empty")

    if max_tokens <= 0:
        return TrimmedHistory([messages[-1]], len(messages) > 1)

    if token_counter(messages) <= max_tokens:
        return TrimmedHistory(messages, False)

    result = trim_messages(
        messages,
        strategy="last",
        token_counter=token_counter,
        max_tokens=max_tokens,
        start_on="human",
        end_on="human",
    )

    if not result or result[-1] != messages[-1]:
        return TrimmedHistory([messages[-1]], True)

    return TrimmedHistory(result, len(result) != len(messages))
