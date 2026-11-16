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
