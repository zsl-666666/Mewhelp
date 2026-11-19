import json

from langchain_core.messages import AIMessageChunk

from app.config import Settings


def make_settings(**overrides) -> Settings:
    values = {
        "llm_base_url": "https://example.com/v1",
        "llm_api_key": "test-key",
        "llm_model": "test-model",
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)


class FakeStreamModel:
    def __init__(self, chunks, error_after=None):
        self.chunks = chunks
        self.error_after = error_after
        self.messages = None

    async def astream(self, messages):
        self.messages = messages
        for index, content in enumerate(self.chunks):
            if self.error_after is not None and index == self.error_after:
                raise RuntimeError("upstream exploded")
            yield AIMessageChunk(content=content)
        if self.error_after is not None and len(self.chunks) == self.error_after:
            raise RuntimeError("upstream exploded")


class FakeBoundModel:
    """`bind_tools()` 的返回值：与父模型共享状态。"""

    def __init__(self, parent):
        self.parent = parent

    async def astream(self, messages):
        self.parent.final_messages = messages
        if self.parent.tool_calls is None:
            for content in self.parent.final_chunks:
                yield AIMessageChunk(content=content)
            return
        for content in self.parent.first_chunks:
            yield AIMessageChunk(content=content)
        tool_call_chunks = [
            {
                "name": call["name"],
                "args": json.dumps(call["args"], ensure_ascii=False),
                "id": call["id"],
                "index": index,
            }
            for index, call in enumerate(self.parent.tool_call_chunks)
        ]
        yield AIMessageChunk(content="", tool_call_chunks=tool_call_chunks)


class FakeToolCallModel:
    def __init__(self, first_chunks=(), final_chunks=(), tool_calls=None):
        self.first_chunks = list(first_chunks)
        self.final_chunks = list(final_chunks)
        self.tool_calls = tool_calls
        self.final_messages = None

    @property
    def tool_call_chunks(self):
        if self.tool_calls is None:
            return []
        if isinstance(self.tool_calls, dict):
            return [self.tool_calls]
        return list(self.tool_calls)

    def bind_tools(self, tools, **kwargs):
        return FakeBoundModel(self)

    async def astream(self, messages):
        self.final_messages = messages
        for content in self.final_chunks:
            yield AIMessageChunk(content=content)


class FakeStructuredRunnable:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.result


class FakeStructuredModel:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []
        self.runnable = None

    def with_structured_output(self, schema, method=None):
        self.calls.append((schema, method))
        self.runnable = FakeStructuredRunnable(self.result, self.error)
        return self.runnable
