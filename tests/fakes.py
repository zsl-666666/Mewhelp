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
