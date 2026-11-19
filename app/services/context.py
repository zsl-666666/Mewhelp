import logging
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

logger = logging.getLogger(__name__)

TokenCounter = Callable[[list[BaseMessage]], int]


@dataclass(frozen=True)
class TrimmedHistory:
    messages: list[BaseMessage]
    trimmed: bool


def repair_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    """删除孤立的工具调用与工具结果，保证 AI tool_call / ToolMessage 成对。

    裁剪历史可能把一对工具调用切成两半；OpenAI 兼容接口遇到不成对的
    工具消息会直接返回 400。这里保持原顺序，只保留成对的消息。
    """
    tool_call_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    # 只保留“所有调用 id 都有工具结果”的 AI 消息；其余 AI 消息整个丢弃。
    matched_ai_ids = {
        call.get("id")
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
        if all(call.get("id") in tool_call_ids for call in message.tool_calls)
        for call in message.tool_calls
    }

    repaired: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            if all(
                call.get("id") in tool_call_ids for call in message.tool_calls
            ):
                repaired.append(message)
        elif isinstance(message, ToolMessage):
            if message.tool_call_id in matched_ai_ids:
                repaired.append(message)
        else:
            repaired.append(message)
    return repaired


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
        logger.warning("latest message exceeds context token budget; sending without history")
        return TrimmedHistory([messages[-1]], len(messages) > 1)

    return TrimmedHistory(result, len(result) != len(messages))
