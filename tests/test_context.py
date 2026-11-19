import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.services.context import trim_history


def char_counter(messages):
    return sum(len(str(message.content)) for message in messages)


def test_under_budget_returns_same_messages():
    messages = [HumanMessage(content="你好"), AIMessage(content="您好")]
    result = trim_history(messages, max_tokens=1000, token_counter=char_counter)
    assert result.messages == messages
    assert result.trimmed is False


def test_over_budget_drops_oldest_and_keeps_latest_user():
    messages = [
        HumanMessage(content="a" * 20),
        AIMessage(content="b" * 20),
        HumanMessage(content="c" * 20),
        AIMessage(content="d" * 20),
        HumanMessage(content="latest"),
    ]
    result = trim_history(messages, max_tokens=40, token_counter=char_counter)
    assert result.trimmed is True
    assert result.messages[-1] == messages[-1]
    assert result.messages[0].type == "human"
    assert len(result.messages) < len(messages)


def test_latest_user_message_is_kept_even_if_over_budget():
    messages = [
        HumanMessage(content="old"),
        AIMessage(content="old reply"),
        HumanMessage(content="x" * 200),
    ]
    result = trim_history(messages, max_tokens=10, token_counter=char_counter)
    assert result.messages == [messages[-1]]
    assert result.trimmed is True


def test_single_over_budget_message_is_not_trimmed():
    messages = [HumanMessage(content="x" * 200)]
    result = trim_history(messages, max_tokens=10, token_counter=char_counter)
    assert result.messages == [messages[-1]]
    assert result.trimmed is False


def test_empty_messages_raises():
    with pytest.raises(ValueError):
        trim_history([], max_tokens=10, token_counter=char_counter)


def test_zero_budget_keeps_only_latest():
    messages = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]
    result = trim_history(messages, max_tokens=0, token_counter=char_counter)
    assert result.messages == [messages[-1]]
    assert result.trimmed is True


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.context import repair_tool_pairs


def test_repair_tool_pairs_drops_orphans():
    orphan_tool = ToolMessage(content="x", tool_call_id="missing")
    orphan_ai = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {}, "id": "lost", "type": "tool_call"}],
    )
    valid_ai = AIMessage(
        content="",
        tool_calls=[{"name": "query_logistics", "args": {}, "id": "ok", "type": "tool_call"}],
    )
    valid_tool = ToolMessage(content="ok", tool_call_id="ok")
    messages = [
        HumanMessage(content="q"),
        orphan_ai,
        orphan_tool,
        valid_ai,
        valid_tool,
        AIMessage(content="a"),
    ]
    repaired = repair_tool_pairs(messages)
    assert repaired == [messages[0], valid_ai, valid_tool, messages[-1]]
