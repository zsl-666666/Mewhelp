from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)

from app.services.chat import extract_text, stream_chat
from tests.fakes import FakeStreamModel, make_settings


async def test_stream_chat_passes_system_prompt_and_trimmed_history():
    model = FakeStreamModel(["好"])
    settings = make_settings()
    history = [HumanMessage(content="hi")]
    async for _ in stream_chat(history, model, settings):
        pass
    assert isinstance(model.messages[0], SystemMessage)
    assert settings.company_name in model.messages[0].content
    assert model.messages[1:] == history


async def test_stream_chat_reports_trimmed_history():
    model = FakeStreamModel(["好"])
    settings = make_settings(context_token_budget=1)
    history = [
        HumanMessage(content="word " * 50),
        AIMessage(content="answer " * 50),
        HumanMessage(content="latest question " * 50),
    ]
    frames = [frame async for frame in stream_chat(history, model, settings)]
    assert frames[-1] == 'data: {"done": true, "trimmed": true}\n\n'


def test_extract_text_handles_string_and_list_content():
    assert extract_text(AIMessageChunk(content="你好")) == "你好"
    assert extract_text(
        AIMessageChunk(content=[{"type": "text", "text": "你"}, {"type": "text", "text": "好"}])
    ) == "你好"
    assert extract_text(AIMessageChunk(content=[])) == ""


async def test_stream_chat_yields_delta_and_done():
    model = FakeStreamModel(["你", "好"])
    settings = make_settings()
    frames = [
        frame
        async for frame in stream_chat(
            [HumanMessage(content="hi")], model, settings
        )
    ]
    assert frames == [
        'data: {"delta": "你"}\n\n',
        'data: {"delta": "好"}\n\n',
        'data: {"done": true, "trimmed": false}\n\n',
    ]


async def test_stream_chat_emits_error_event_on_mid_stream_failure():
    model = FakeStreamModel(["你", "好"], error_after=1)
    settings = make_settings()
    frames = [
        frame
        async for frame in stream_chat(
            [HumanMessage(content="hi")], model, settings
        )
    ]
    assert 'data: {"delta": "你"}\n\n' in frames
    assert frames[-1].startswith("event: error\n")
