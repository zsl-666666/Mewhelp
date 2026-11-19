# Chapter 02 Function Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有客服聊天接入单轮 Function Calling：模型选择五个工具之一，后端执行并回灌结果，最终回答继续 SSE 流式输出，整轮对话（含工具调用与结果）落到 MySQL。

**Architecture:** 服务端管历史。`POST /api/chat` 只收 `{user_id, conversation_id?, message}`；路由先建会话/落用户消息，再进入 SSE。`services/chat.py` 负责裁剪、绑定工具、执行、回灌、二次调用与落库；`repositories/` 封装 SQL；`tools/` 只定义工具契约与 mock/DB 查询；最终回答使用未绑定工具的 model 强制收敛。

**Tech Stack:** Python 3.13、uv、FastAPI、LangChain `@tool` / `bind_tools`、SQLAlchemy 2.x、PyMySQL、MySQL 8.0（Docker）、SQLite（仅单测）。

**Spec:** `docs/superpowers/specs/2026-11-19-ch02-function-calling-design.md`

## Global Constraints

- 数据库以 `db/schema.sql` 为唯一事实来源；MySQL 表已由该 DDL 创建，禁止用 `create_all` 建 MySQL 业务表。
- 统一 `ENGINE=InnoDB`、`CHARSET=utf8mb4`、`COLLATE=utf8mb4_unicode_ci`。
- `/api/chat` 请求体固定为 `{user_id, conversation_id?, message}`；旧 `messages` 契约删除。
- 只做单轮工具调用：模型返回多个 `tool_calls` 时只执行第一个，其余丢弃并记录 warning。
- 每次工具调用必须以 `completed` / `timeout` / `failed` 之一收尾；不能只发 `running`。
- `create_ticket` 不可重试；成功后同事务把 `conversations.status` 改为 `已转人工`。
- `TOOL_RESULT_MAX_CHARS = 4000`；当前轮模型拿完整结果，持久化只存截断版本并标记 `truncated`。
- `query_faq` 只做字面 LIKE；“邮费”查不到“运费”是 known limitation，不是 bug；不得补种子“邮费”或让模型改写关键词。
- 真实密码只在本机 `.env`，不得进入代码、测试、日志、Compose 或 Git。
- 聊天页改造走 Vibe Coding 例外：直接改 `app/static/index.html`，不写前端自动化测试，按浏览器验收。
- 技术选型定死；真实模型不支持工具调用时停下询问，不擅自换模型/框架。

## Review Focus

1. **历史裁剪切断工具对**：`trim_history` 可能留下孤立 `ToolMessage` 或缺失结果的 `AIMessage(tool_calls=...)`，OpenAI 兼容接口会 400。Task 7 用 `repair_tool_pairs` 修复并测试。
2. **超时/重试后没有终点帧**：重试耗尽必须发 `timeout` 或 `failed`，不能停在 `running`/`retrying`。Task 4 与 Task 8 覆盖事件流与 SSE 帧。
3. **工具结果过长**：超过 4000 字符时持久化必须截断且保留 `original_chars`，当前轮模型仍应收到完整结果。Task 7 覆盖。
4. **`create_ticket` 重试/重复工单**：注册为不可重试；失败只执行一次；成功后会话状态必须变为 `已转人工`。Task 4 与 Task 5 覆盖。
5. **FAQ 字面 LIKE 的 wildcard 与 known limitation**：`%` / `_` 必须转义；“邮费”漏召回按预期返回空结果，不视为失败。Task 2 与 Task 5 覆盖。

---

### Task 1: ORM 模型与 SQLite 测试基础设施

**Files:**
- Create: `app/models.py`
- Modify: `tests/conftest.py`（在现有 `client` fixture 后追加 `session_factory`）
- Test: `tests/test_models.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: `db/schema.sql`、`app/db.py`
- Produces:
  - `Base`、`Faq`、`Conversation`、`Message`、`Ticket`
  - `session_factory` fixture：`sessionmaker[Session]`，绑定每个测试独立的临时 SQLite 文件，`check_same_thread=False`、`expire_on_commit=False`
  - `Base.metadata.create_all(engine)` 可在 SQLite 建出带 CHECK 约束的测试表

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Conversation, Faq, Message, Ticket


def test_models_roundtrip_and_nullable_tool_fields(session_factory):
    with session_factory() as session:
        convo = Conversation(user_id="u1", status="进行中")
        session.add(convo)
        session.commit()
        session.refresh(convo)

        message = Message(
            conversation_id=convo.id,
            role="assistant",
            content=None,
            tool_calls=[{
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        faq = Faq(
            question="退货政策是什么？",
            answer="签收后 7 天内支持无理由退货。",
            category="售后",
        )
        ticket = Ticket(
            ticket_no="T20260701008",
            conversation_id=convo.id,
            description="耳机坏了",
            ticket_type="售后",
        )
        session.add_all([message, faq, ticket])
        session.commit()
        session.refresh(faq)

        assert convo.status == "进行中"
        assert message.content is None
        assert message.tool_calls[0]["id"] == "call_1"
        assert faq.updated_at is not None
        assert ticket.status == "待处理"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Conversation(user_id="u1", status="active"),
        lambda: Message(conversation_id=1, role="system", content="x"),
        lambda: Ticket(
            ticket_no="T1",
            conversation_id=1,
            description="x",
            ticket_type="退款",
        ),
    ],
)
def test_models_reject_invalid_enum_values(session_factory, factory):
    with session_factory() as session:
        session.add(factory())
        with pytest.raises(IntegrityError):
            session.commit()
```

```python
# tests/test_schema.py
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def test_schema_has_four_tables_in_order_and_target_collation():
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    assert sql.count("CREATE TABLE") == 4
    positions = [
        sql.index(f"CREATE TABLE {name}")
        for name in ("conversations", "messages", "faq", "tickets")
    ]
    assert positions == sorted(positions)
    table_suffix = "ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
    assert sql.count(table_suffix) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py tests/test_schema.py -v`

Expected: FAIL，`ModuleNotFoundError: app.models`；或 `fixture 'session_factory' not found`。

- [ ] **Step 3: Implement models and fixture**

在 `app/models.py` 定义 `Base(DeclarativeBase)` 及四张表，字段严格对齐 spec 7.1–7.4：

- `id` / `conversation_id` 使用 `BigInteger().with_variant(Integer, "sqlite")`。
- `faq`：`question String(512)`、`answer Text`、`category String(64)`、`created_at` / `updated_at`；`category` 建索引。
- `conversations`：`user_id String(64)` 建索引、`status String(16)`、`created_at` / `updated_at`。
- `messages`：`role String(16)`、`content Text nullable=True`、`tool_calls JSON nullable=True`、`tool_call_id String(64) nullable=True`；`conversation_id` 外键、索引。
- `tickets`：`ticket_no String(32)` 主键、`conversation_id` 外键/索引、`description Text`、`ticket_type String(16)`、`status String(16)`、`created_at`。
- 用 `CheckConstraint` 固定：`conversations.status IN ('进行中','已转人工','已结束')`、`messages.role IN ('user','assistant','tool')`、`tickets.ticket_type IN ('售后','投诉','咨询')`、`tickets.status IN ('待处理','已处理')`。
- `created_at` 用 `server_default=func.now()`；`updated_at` 用 `server_default=func.now(), onupdate=func.now()`。
- 外键不加 `ondelete`（MySQL DDL 是默认 RESTRICT）。
- 关系只建 `Conversation.messages` / `Conversation.tickets`，不配置删除级联。

在 `tests/conftest.py` 追加：

```python
@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        engine.dispose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py tests/test_schema.py -v`

Expected: PASS（模型用例 + schema 用例全绿）。

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/conftest.py tests/test_models.py tests/test_schema.py
git commit -m "feat: add chapter 2 ORM models" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: 会话/FAQ/工单仓储与 FAQ 种子

**Files:**
- Create: `app/repositories/__init__.py`
- Create: `app/repositories/conversations.py`
- Create: `app/repositories/faq.py`
- Create: `app/repositories/tickets.py`
- Create: `app/seed_data.py`
- Test: `tests/test_repositories.py`, `tests/test_seed.py`

**Interfaces:**
- Consumes: Task 1 的模型与 `session_factory`
- Produces:
  - `create_conversation(session: Session, user_id: str) -> Conversation`
  - `get_conversation(session: Session, conversation_id: int) -> Conversation | None`
  - `append_message(session: Session, conversation_id: int, *, role: str, content: str | None = None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> Message`
  - `load_history(session: Session, conversation_id: int) -> list[BaseMessage]`
  - `set_conversation_status(session: Session, conversation_id: int, status: str) -> None`
  - `search_faq(session: Session, keyword: str, limit: int = 3) -> list[Faq]`
  - `FAQ_SEED: tuple[dict[str, str], ...]`
  - `seed_faq(session: Session) -> int`（表非空返回 0）
  - `create_ticket_record(session: Session, conversation_id: int, description: str, ticket_type: str) -> Ticket`
  - `get_ticket(session: Session, ticket_no: str) -> Ticket | None`

- [ ] **Step 1: Write the failing repository tests**

```python
# tests/test_repositories.py
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.repositories.conversations import (
    append_message,
    create_conversation,
    load_history,
)
from app.repositories.faq import search_faq
from app.repositories.tickets import create_ticket_record
from app.models import Faq


def test_load_history_rebuilds_tool_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")
        append_message(
            session,
            convo.id,
            role="assistant",
            content="",
            tool_calls=[{
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            }],
        )
        envelope = json.dumps({
            "tool": "query_logistics",
            "arguments": {"order_id": "1001"},
            "ok": True,
            "result": '{"ok": true, "data": {"status": "运输中"}}',
            "truncated": False,
            "original_chars": 40,
            "attempts": 1,
            "duration_ms": 12,
        }, ensure_ascii=False)
        append_message(
            session,
            convo.id,
            role="tool",
            content=envelope,
            tool_call_id="call_1",
        )
        append_message(session, convo.id, role="assistant", content="预计明天送达。")

        history = load_history(session, convo.id)

    assert [type(message).__name__ for message in history] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage"
    ]
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[1].tool_calls[0]["id"] == "call_1"
    assert isinstance(history[2], ToolMessage)
    assert history[2].tool_call_id == "call_1"
    assert history[2].content == '{"ok": true, "data": {"status": "运输中"}}'


def test_search_faq_escapes_like_wildcards(session_factory):
    with session_factory() as session:
        session.add_all([
            Faq(question="含 100% 纯棉", answer="是", category="商品"),
            Faq(question="100 元优惠券", answer="是", category="活动"),
        ])
        session.commit()
        rows = search_faq(session, "100%")

    assert [row.question for row in rows] == ["含 100% 纯棉"]


def test_create_ticket_record_updates_conversation_status(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        ticket = create_ticket_record(session, convo.id, "耳机坏了", "售后")
        session.refresh(convo)

    assert ticket.status == "待处理"
    assert ticket.ticket_type == "售后"
    assert ticket.ticket_no.startswith("T")
    assert len(ticket.ticket_no) == 19
    assert convo.status == "已转人工"


def test_create_ticket_record_rejects_invalid_type_and_missing_conversation(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        with pytest.raises(ValueError):
            create_ticket_record(session, convo.id, "x", "退款")
        with pytest.raises(ValueError):
            create_ticket_record(session, 99999, "x", "售后")
```

```python
# tests/test_seed.py
from sqlalchemy import select

from app.models import Faq
from app.repositories.faq import FAQ_SEED, seed_faq


def test_seed_faq_is_idempotent_and_contains_known_limitation(session_factory):
    with session_factory() as session:
        assert seed_faq(session) == len(FAQ_SEED)
        assert seed_faq(session) == 0
        rows = session.scalars(select(Faq)).all()

    assert any("退货政策" in row.question for row in rows)
    assert all("邮费" not in (row.question + row.answer + row.category) for row in rows)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_repositories.py tests/test_seed.py -v`

Expected: FAIL，`ModuleNotFoundError: app.repositories` 或缺少 `FAQ_SEED`。

- [ ] **Step 3: Implement repositories and seed data**

- `conversations.py`：`create_conversation` 写 `status="进行中"`；`append_message` 写一条并 commit/refresh；`load_history` 按 `Message.id` 升序重建：
  - `user` -> `HumanMessage(content=content or "")`
  - `assistant` -> `AIMessage(content=content or "", tool_calls=tool_calls or [])`
  - `tool` -> 解析 `content` JSON envelope，`ToolMessage(content=payload["result"], tool_call_id=tool_call_id)`；解析失败则用原 content。
- `faq.py`：`search_faq` 用 `or_(Faq.question.contains(keyword, autoescape=True), Faq.answer.contains(keyword, autoescape=True))`，`order_by(Faq.id)`，`limit(limit)`；`seed_faq` 先 `select(func.count()).select_from(Faq)`，非空返回 0，否则插入 `FAQ_SEED` 并 commit。
- `app/seed_data.py`：10 条 FAQ，必须包含“退货政策是什么？”与“运费怎么计算？”，任何字段不得出现“邮费”。
- `tickets.py`：`create_ticket_record` 校验 `ticket_type in {"售后","投诉","咨询"}`；加载 conversation，不存在抛 `ValueError`；生成 `T{YYYYMMDDHHMMSS}{4位HEX}`；写 `Ticket(status="待处理")`，把 conversation.status 设为 `"已转人工"`，同一 transaction commit。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_repositories.py tests/test_seed.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/repositories app/seed_data.py tests/test_repositories.py tests/test_seed.py
git commit -m "feat: add repositories and FAQ seed data" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: 新请求模型与工具配置

**Files:**
- Modify: `app/schemas.py:5-28`
- Modify: `app/config.py:14-26`
- Test: `tests/test_schemas.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `ChatRequest(user_id="demo-user", conversation_id=None, message: str)`
  - `Settings.tool_timeout_seconds=5.0`、`tool_max_retries=2`、`tool_retry_delay_seconds=0.2`、`tool_result_max_chars=4000`

- [ ] **Step 1: Write the failing schema/config tests**

```python
# tests/test_schemas.py（替换旧的 ChatMessage/ChatRequest 用例）
import pytest
from pydantic import ValidationError

from app.schemas import ChatRequest


def test_chat_request_defaults():
    request = ChatRequest(message="你好")
    assert request.user_id == "demo-user"
    assert request.conversation_id is None


def test_chat_request_accepts_conversation_id():
    request = ChatRequest(user_id="u1", conversation_id=12, message="继续")
    assert request.conversation_id == 12


@pytest.mark.parametrize("payload", [
    {"message": "   "},
    {"user_id": "   ", "message": "你好"},
    {"user_id": "u1", "conversation_id": 0, "message": "你好"},
])
def test_chat_request_rejects_invalid(payload):
    with pytest.raises(ValidationError):
        ChatRequest(**payload)
```

```python
# tests/test_config.py 追加
def test_tool_settings_defaults(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    settings = Settings(_env_file=None)
    assert settings.tool_timeout_seconds == 5.0
    assert settings.tool_max_retries == 2
    assert settings.tool_retry_delay_seconds == 0.2
    assert settings.tool_result_max_chars == 4000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py tests/test_config.py -v`

Expected: FAIL，`ChatRequest` 仍要求 `messages`；`Settings` 缺少 `tool_*` 字段。

- [ ] **Step 3: Implement request model and settings**

- `app/schemas.py`：删除 `ChatMessage` 与旧 `ChatRequest`，新增上述 `ChatRequest`；`user_id` 1–64 字符且非空白，`conversation_id` 为正整数，`message` 非空白。保留 `AfterSales` 模型不变。
- `app/config.py`：新增四个字段；用 `Field(gt=0)` 约束 timeout/result_max_chars，`Field(ge=0)` 约束 retries/delay。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py tests/test_config.py -v`

Expected: PASS；`AfterSales` 相关用例仍通过。

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py app/config.py tests/test_schemas.py tests/test_config.py
git commit -m "feat: add server-managed chat request and tool settings" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: 工具注册表、执行事件流与 mock 电商工具

**Files:**
- Create: `app/tools/__init__.py`
- Create: `app/tools/registry.py`
- Create: `app/tools/ecommerce.py`
- Test: `tests/test_registry.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: LangChain `BaseTool` / `@tool` / `InjectedToolArg`
- Produces:
  - `ToolSpec(tool: BaseTool, label: str, retryable: bool)`
  - `ToolExecutionResult(ok: bool, content: str, error: str | None, error_code: str | None, retryable: bool, attempts: int, duration_ms: int)`
  - `ToolExecutionEvent(status, tool_call_id, name, label, arguments, attempt=0, max_attempts=0, error_code=None, error=None, result=None)`
  - `ToolRegistry.register(tool, *, label, retryable=True)`
  - `ToolRegistry.get(name) -> ToolSpec | None`
  - `ToolRegistry.bindable_tools() -> list[BaseTool]`
  - `ToolRegistry.execute_stream(tool_call, *, injected_args=None, timeout_seconds, max_retries, retry_delay_seconds) -> AsyncIterator[ToolExecutionEvent]`
  - `ToolRegistry.execute(...) -> ToolExecutionResult`
  - `query_order` / `query_product` / `query_logistics` 三个 `@tool`

- [ ] **Step 1: Write the failing registry tests**

```python
# tests/test_registry.py
import asyncio
import json
from typing import Annotated, Literal

import pytest
from langchain_core.tools import InjectedToolArg, tool

from app.tools.registry import ToolRegistry


@tool
def echo(text: str) -> str:
    """回显文本。"""
    return json.dumps({"ok": True, "data": {"text": text}}, ensure_ascii=False)


@tool
async def slow(text: str) -> str:
    """故意慢。"""
    await asyncio.sleep(0.2)
    return json.dumps({"ok": True, "data": {"text": text}}, ensure_ascii=False)


@tool
def boom(text: str) -> str:
    """故意失败。"""
    raise RuntimeError("boom")


@tool
def needs_int(value: int) -> str:
    """需要整数。"""
    return json.dumps({"ok": True, "data": {"value": value}}, ensure_ascii=False)


@tool
def with_conversation(
    text: str,
    conversation_id: Annotated[int, InjectedToolArg],
) -> str:
    """需要注入会话 id。"""
    return json.dumps({"ok": True, "data": {"conversation_id": conversation_id}}, ensure_ascii=False)


def build_registry(tool_obj, *, retryable=True):
    registry = ToolRegistry()
    registry.register(tool_obj, label="测试工具", retryable=retryable)
    return registry


async def test_execute_stream_success():
    registry = build_registry(echo)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "echo", "args": {"text": "hi"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "completed"]
    assert events[-1].result.ok is True
    assert json.loads(events[-1].result.content)["data"]["text"] == "hi"


async def test_execute_stream_validation_error_does_not_retry():
    registry = build_registry(needs_int)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "needs_int", "args": {"value": "bad"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "validation_error"
    assert events[-1].result.attempts == 0


async def test_execute_stream_timeout_retries_then_timeout():
    registry = build_registry(slow)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "slow", "args": {"text": "hi"}},
            timeout_seconds=0.01,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == [
        "started", "retrying", "retrying", "timeout"
    ]
    assert events[-1].result.error_code == "timeout"
    assert events[-1].result.attempts == 3


async def test_execute_stream_non_retryable_fails_once():
    registry = build_registry(boom, retryable=False)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "boom", "args": {"text": "hi"}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "execution_error"
    assert events[-1].result.attempts == 1


async def test_unknown_tool_fails_without_retry():
    registry = build_registry(echo)
    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "missing", "args": {}},
            timeout_seconds=1,
            max_retries=2,
            retry_delay_seconds=0,
        )
    ]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].result.error_code == "unknown_tool"
    assert events[-1].result.attempts == 0


async def test_injected_arg_hidden_from_model_schema():
    registry = build_registry(with_conversation)
    schema = registry.bindable_tools()[0].tool_call_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]

    events = [
        event async for event in registry.execute_stream(
            {"id": "call_1", "name": "with_conversation", "args": {"text": "hi"}},
            injected_args={"conversation_id": 7},
            timeout_seconds=1,
            max_retries=0,
            retry_delay_seconds=0,
        )
    ]
    assert events[-1].result.ok is True
    assert events[-1].arguments == {"text": "hi"}
```

```python
# tests/test_tools.py（先写 mock 工具部分）
import json

from app.tools.ecommerce import query_logistics, query_order, query_product


def test_ecommerce_tools_are_deterministic():
    first = query_order.invoke({"order_id": "1001"})
    second = query_order.invoke({"order_id": "1001"})
    assert first == second
    assert json.loads(first)["ok"] is True

    logistics = json.loads(query_logistics.invoke({"order_id": "1001"}))
    product = json.loads(query_product.invoke({"product_id": "P100"}))
    assert logistics["data"]["order_id"] == "1001"
    assert product["data"]["product_id"] == "P100"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py tests/test_tools.py -v`

Expected: FAIL，`ModuleNotFoundError: app.tools.registry` / `app.tools.ecommerce`。

- [ ] **Step 3: Implement registry and mock tools**

- `registry.py`：按 Interfaces 实现 dataclasses 与 `ToolRegistry`。执行流程：
  1. 发 `started`（`attempt=1`，`max_attempts=1+max_retries` 或 1）。
  2. 未知工具：直接 `failed`，`error_code="unknown_tool"`、`retryable=False`、`attempts=0`。
  3. 合并 `arguments` 与 `injected_args`，只把 `args_schema.model_fields` 中存在的注入键合并；`model_validate` 失败 -> `failed`，`validation_error`、`attempts=0`。
  4. 每次尝试 `asyncio.wait_for(tool.ainvoke(validated.model_dump()), timeout=timeout_seconds)`。
  5. 超时 -> 可重试则发 `retrying`（`attempt` 为下一次），否则最终 `timeout`；异常 -> 同理，最终 `failed` / `execution_error`。
  6. 成功 -> `completed`，`content` 为字符串或 JSON dump。
  7. `duration_ms` 覆盖整个重试过程；`execute` 消费事件流并返回最终 `result`。
- 错误文案脱敏：timeout “工具执行超时”；validation “参数校验失败”；execution “工具执行失败”；unknown “未知工具: <name>”。不要返回堆栈。
- `ecommerce.py`：三个 `@tool`，参数用 `Annotated[str, Field(min_length=1)]`；用 `random.Random(f"order:{id}")` / `f"product:{id}"` 生成确定性 mock，返回 `json.dumps({"ok": True, "data": ...}, ensure_ascii=False)`。物流结果至少包含 `order_id`、`carrier`、`tracking_no`、`status`、`latest_event`、`estimated_delivery`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py tests/test_tools.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/tools tests/test_registry.py tests/test_tools.py
git commit -m "feat: add tool registry and mock commerce tools" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: FAQ/工单工具与注册表工厂

**Files:**
- Create: `app/tools/knowledge.py`
- Create: `app/tools/tickets.py`
- Modify: `app/tools/__init__.py`
- Test: `tests/test_tools.py`（追加 FAQ/工单用例）

**Interfaces:**
- Consumes: Task 2 的 `search_faq` / `create_ticket_record`，Task 4 的 `ToolRegistry`
- Produces:
  - `build_query_faq_tool(session_factory) -> BaseTool`
  - `build_create_ticket_tool(session_factory) -> BaseTool`
  - `build_tool_registry(session_factory) -> ToolRegistry`（五个工具，标签：查订单/查商品/查物流/查 FAQ/转人工）

- [ ] **Step 1: Write the failing tool tests**

```python
# tests/test_tools.py 追加
import json

from app.repositories.faq import seed_faq
from app.tools import build_tool_registry
from app.tools.knowledge import build_query_faq_tool
from app.tools.tickets import build_create_ticket_tool


def test_query_faq_tool_hit_and_miss(session_factory):
    with session_factory() as session:
        seed_faq(session)

    query_faq = build_query_faq_tool(session_factory)
    hit = json.loads(query_faq.invoke({"keyword": "退货政策"}))
    miss = json.loads(query_faq.invoke({"keyword": "邮费"}))

    assert hit["ok"] is True
    assert any("退货政策" in item["question"] for item in hit["data"])
    assert miss["ok"] is True
    assert miss["data"] == []
    assert miss["message"] == "未找到相关 FAQ"


def test_create_ticket_tool_writes_and_updates_status(session_factory):
    from app.repositories.conversations import create_conversation

    with session_factory() as session:
        convo = create_conversation(session, "u1")

    create_ticket = build_create_ticket_tool(session_factory)
    result = json.loads(create_ticket.invoke({
        "description": "耳机坏了，需要人工处理",
        "ticket_type": "售后",
        "conversation_id": convo.id,
    }))

    assert result["ok"] is True
    assert result["data"]["ticket_type"] == "售后"
    with session_factory() as session:
        from app.models import Conversation, Ticket
        assert session.get(Conversation, convo.id).status == "已转人工"
        assert session.get(Ticket, result["data"]["ticket_no"]).status == "待处理"


def test_build_tool_registry_has_five_tools_and_labels(session_factory):
    registry = build_tool_registry(session_factory)
    assert [tool.name for tool in registry.bindable_tools()] == [
        "query_order", "query_product", "query_logistics", "query_faq", "create_ticket"
    ]
    assert registry.get("query_logistics").label == "查物流"
    assert registry.get("create_ticket").retryable is False


def test_create_ticket_schema_hides_injected_conversation_id(session_factory):
    registry = build_tool_registry(session_factory)
    schema = registry.get("create_ticket").tool.tool_call_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]
    assert "ticket_type" in schema["properties"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`

Expected: FAIL，缺少 `build_query_faq_tool` / `build_create_ticket_tool` / `build_tool_registry`。

- [ ] **Step 3: Implement tool factories and registry builder**

- `knowledge.py`：闭包捕获 `session_factory`；`@tool` 函数只暴露 `keyword: Annotated[str, Field(min_length=1)]` 与 `limit: Annotated[int, Field(ge=1, le=10)] = 3`。docstring 明确“使用用户原话关键词，不要改写同义词”。命中返回 `{"ok": true, "data": [{"question","answer","category"}]}`；未命中返回 spec 9.3 的空结果。
- `tickets.py`：闭包捕获 `session_factory`；签名 `create_ticket(description, ticket_type: Literal["售后","投诉","咨询"], conversation_id: Annotated[int, InjectedToolArg])`。调用 `create_ticket_record`，返回 `{"ok": true, "data": {"ticket_no", "ticket_type", "status"}}`。
- `app/tools/__init__.py`：`build_tool_registry` 注册三个 mock 工具、`build_query_faq_tool`（label `查 FAQ`）、`build_create_ticket_tool`（label `转人工`，`retryable=False`）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_registry.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/tools tests/test_tools.py
git commit -m "feat: add FAQ and ticket tools" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: 工具使用 System Prompt

**Files:**
- Modify: `app/prompts/customer_service.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: 无
- Produces: `TOOL_USAGE_RULES: str`（chat service 用 `SystemMessage(content=TOOL_USAGE_RULES)` 追加）

- [ ] **Step 1: Write the failing prompt test**

```python
# tests/test_prompts.py 追加
from app.prompts.customer_service import TOOL_USAGE_RULES


def test_tool_usage_rules_pin_single_call_and_faq_keyword():
    assert "只调用一个工具" in TOOL_USAGE_RULES
    assert "邮费" in TOOL_USAGE_RULES
    assert "运费" in TOOL_USAGE_RULES
    assert "不要改写" in TOOL_USAGE_RULES
    assert "create_ticket" in TOOL_USAGE_RULES
    for value in ("售后", "投诉", "咨询"):
        assert value in TOOL_USAGE_RULES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py::test_tool_usage_rules_pin_single_call_and_faq_keyword -v`

Expected: FAIL，`ImportError: TOOL_USAGE_RULES`。

- [ ] **Step 3: Implement the rules**

在 `app/prompts/customer_service.py` 新增 `TOOL_USAGE_RULES`，至少包含：

- 需要订单/商品/物流/FAQ/工单信息时优先调用对应工具，不编造。
- 每轮只调用一个工具；拿到结果后直接给最终回答，不再调用。
- FAQ 必须使用用户原话关键词，不要改写同义词；明确举“邮费”不要改成“运费”。
- 工具空结果/错误时如实说明并建议转人工，不编造。
- 只有用户明确要求人工/投诉/建工单时才调用 `create_ticket`；`ticket_type` 只能是 `售后` / `投诉` / `咨询`。
- 不向用户展示工具名、JSON 或内部参数。

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/prompts/customer_service.py tests/test_prompts.py
git commit -m "feat: add tool usage system rules" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 聊天编排、工具落库与工具对修复

**Files:**
- Modify: `app/services/chat.py`（整体重写）
- Modify: `app/services/context.py`（新增 `repair_tool_pairs`）
- Modify: `tests/fakes.py`（新增 `FakeToolCallModel` / `FakeBoundModel`）
- Test: `tests/test_chat.py`（整体重写）、`tests/test_context.py`（追加）

**Interfaces:**
- Consumes: Task 2 仓储、Task 4/5 registry、Task 6 `TOOL_USAGE_RULES`
- Produces:
  - `stream_chat(*, conversation_id: int, user_id: str, session_factory, registry: ToolRegistry, model: BaseChatModel, settings: Settings) -> AsyncIterator[str]`
  - `build_tool_message_content(tool_call: dict[str, Any], result: ToolExecutionResult, max_chars: int) -> str`
  - `repair_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]`
  - `tests/fakes.py` 的 `FakeToolCallModel(first_chunks=(), final_chunks=(), tool_calls=None)`：`bind_tools()` 返回 `FakeBoundModel`；`FakeBoundModel.astream()` 在 `tool_calls is None` 时产出 `final_chunks`，否则产出 `first_chunks` + tool-call chunk；`FakeBoundModel.astream()` 与 `FakeToolCallModel.astream()` 都把收到的 messages 记录到父对象的 `final_messages`（后一次调用覆盖前一次，工具路径最终指向第二次模型调用）

- [ ] **Step 1: Write the failing chat/context tests**

```python
# tests/test_context.py 追加
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
```

```python
# tests/test_chat.py（关键用例）
import json
from langchain_core.messages import ToolMessage
from sqlalchemy import select

from app.models import Message
from app.repositories.conversations import append_message, create_conversation
from app.services.chat import build_tool_message_content, stream_chat
from app.tools.ecommerce import query_logistics
from app.tools.registry import ToolExecutionResult, ToolRegistry
from tests.fakes import FakeStreamModel, FakeToolCallModel, make_settings


def test_build_tool_message_content_truncates_at_limit():
    tool_call = {
        "id": "call_1",
        "name": "query_logistics",
        "args": {"order_id": "1001"},
    }
    result = ToolExecutionResult(
        ok=True,
        content="x" * 5000,
        error=None,
        error_code=None,
        retryable=False,
        attempts=1,
        duration_ms=12,
    )
    payload = json.loads(build_tool_message_content(tool_call, result, max_chars=4000))
    assert payload["truncated"] is True
    assert payload["original_chars"] == 5000
    assert len(payload["result"]) == 4000
    assert payload["tool"] == "query_logistics"
    assert payload["arguments"] == {"order_id": "1001"}


def test_build_tool_message_content_failure_envelope():
    tool_call = {
        "id": "call_1",
        "name": "query_logistics",
        "args": {"order_id": "1001"},
    }
    result = ToolExecutionResult(
        ok=False,
        content='{"ok": false, "error": "工具执行超时"}',
        error="工具执行超时",
        error_code="timeout",
        retryable=True,
        attempts=3,
        duration_ms=5003,
    )
    payload = json.loads(build_tool_message_content(tool_call, result, max_chars=4000))
    assert payload["ok"] is False
    assert payload["error_code"] == "timeout"
    assert payload["retryable"] is True
    assert payload["attempts"] == 3
    assert payload["duration_ms"] == 5003


async def test_stream_chat_no_tool_path_persists_two_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="你好")

    model = FakeToolCallModel(final_chunks=["你", "好"])
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=ToolRegistry(),
            model=model,
            settings=make_settings(),
        )
    ]

    assert frames[0] == 'event: meta\ndata: {"conversation_id": 1, "user_id": "u1"}\n\n'
    assert any('"delta": "你"' in frame for frame in frames)
    assert any('"done": true' in frame and '"tool_calls": []' in frame for frame in frames)
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[-1].content == "你好"


async def test_stream_chat_tool_path_persists_four_messages(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="订单 1001 的物流到哪了")

    registry = ToolRegistry()
    registry.register(query_logistics, label="查物流")
    model = FakeToolCallModel(
        tool_calls=[{
            "name": "query_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["预计明天送达。"],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert any('"status": "running"' in frame for frame in frames)
    assert any('"status": "completed"' in frame for frame in frames)
    assert any('"tool_calls": ["query_logistics"]' in frame for frame in frames)
    tool_messages = [m for m in model.final_messages if isinstance(m, ToolMessage)]
    assert tool_messages and json.loads(tool_messages[0].content)["ok"] is True
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[1].tool_calls[0]["id"] == "call_1"
    assert rows[2].tool_call_id == "call_1"
    envelope = json.loads(rows[2].content)
    assert envelope["tool"] == "query_logistics"
    assert envelope["arguments"] == {"order_id": "1001"}
    assert envelope["ok"] is True


async def test_stream_chat_only_first_tool_call(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="查订单和物流")

    registry = ToolRegistry()
    registry.register(query_logistics, label="查物流")
    model = FakeToolCallModel(
        tool_calls=[
            {
                "name": "query_logistics",
                "args": {"order_id": "1001"},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "query_order",
                "args": {"order_id": "1001"},
                "id": "call_2",
                "type": "tool_call",
            },
        ],
        final_chunks=["已为您查询。"],
    )
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=registry,
            model=model,
            settings=make_settings(),
        )
    ]

    assert any('"tool_calls": ["query_logistics"]' in frame for frame in frames)
    with session_factory() as session:
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert len(rows[1].tool_calls) == 1
    assert rows[1].tool_calls[0]["id"] == "call_1"


async def test_stream_chat_model_error_emits_error_event(session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="你好")

    model = FakeStreamModel(["你"], error_after=1)
    frames = [
        frame async for frame in stream_chat(
            conversation_id=convo.id,
            user_id="u1",
            session_factory=session_factory,
            registry=ToolRegistry(),
            model=model,
            settings=make_settings(),
        )
    ]

    assert any("event: error" in frame for frame in frames)
    assert any('"delta": "你"' in frame for frame in frames)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chat.py tests/test_context.py -v`

Expected: FAIL，新签名/`repair_tool_pairs` / fakes 不存在；旧 `stream_chat(history, model, settings)` 用例失败。

- [ ] **Step 3: Implement orchestration**

- `context.py`：新增 `repair_tool_pairs`。收集每个 `AIMessage.tool_calls` 的 id；保留满足“所有 id 都有对应 ToolMessage”的 AI 消息；删除没有对应 AI 调用的 ToolMessage；保持原顺序。
- `chat.py`：实现下述顺序。
  1. `load_history` 后 `trim_history`，再 `repair_tool_pairs`；记录 `trimmed`。
  2. 拼一个 `SystemMessage`（客服 Prompt 文本 + `"\n\n"` + `TOOL_USAGE_RULES`）+ 历史；不要拆成两条 system 消息，避免部分 OpenAI 兼容端点只接受单条 system。
  3. 先 yield `meta` 帧。
  4. 第一次调用：若 `registry.bindable_tools()` 为空，直接 `model.astream(messages)`；否则 `model.bind_tools(registry.bindable_tools()).astream(messages)`。聚合 `AIMessageChunk`，文本实时 `delta`。
  5. 无 tool_calls：把累计文本作为 assistant 消息落库（空文本不落），yield `done`，`tool_calls=[]`。
  6. 有 tool_calls：只取第一个，`log.warning` 丢弃其余；落库 assistant 工具调用行（`content=累计文本 or None`，`tool_calls=[tool_call]`）；把 `AIMessage(content=累计文本, tool_calls=[tool_call])` 加入消息。
  7. `async for event in registry.execute_stream(...)`：把 `started` 映射为 `running`，`retrying` / `completed` / `timeout` / `failed` 原样映射，按 8.1 发 `tool` 帧（字段与 spec 8.1 完全一致）；最终事件拿 `result`；用 `build_tool_message_content` 落库 tool 行；把 `ToolMessage(content=result.content, tool_call_id=id)` 加入消息。
  8. 第二次调用未绑定工具的 `model.astream(messages)`，逐 delta 输出，累计最终回答；落库 assistant 最终行；yield `done`，`tool_calls=[tool_call["name"]]`。
  9. 异常：记录日志；尽量落库已累计文本；yield `event: error`。
- `build_tool_message_content`：按 spec 7.6 生成 JSON envelope；`result` 截断到 `max_chars`，`truncated`、`original_chars`、`attempts`、`duration_ms` 必填；失败时加 `error_code`、`retryable`。
- 落库统一走 `await asyncio.to_thread(...)`，避免阻塞事件循环。
- `tests/fakes.py`：新增 `FakeBoundModel` 与 `FakeToolCallModel`；tool-call chunk 用 `AIMessageChunk(content="", tool_call_chunks=[{"name": ..., "args": json.dumps(args), "id": ..., "index": 0}])`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chat.py tests/test_context.py -v`

Expected: PASS；旧 `test_chat.py` 中不再适用的无状态用例已替换。

- [ ] **Step 5: Commit**

```bash
git add app/services/chat.py app/services/context.py tests/fakes.py tests/test_chat.py tests/test_context.py
git commit -m "feat: add single-round tool orchestration and persistence" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: `/api/chat` 路由、会话准备与 SSE 集成

**Files:**
- Modify: `app/db.py`（新增 `get_db_session_factory`）
- Modify: `app/main.py:40-55`（替换 chat 路由）
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: Task 2 会话仓储、Task 5 `build_tool_registry`、Task 7 `stream_chat`
- Produces:
  - `get_db_session_factory() -> sessionmaker[Session]`
  - `get_tool_registry(session_factory=Depends(get_db_session_factory)) -> ToolRegistry`
  - `POST /api/chat`：首次建会话、未知 id 404、落 user 消息、返回 `StreamingResponse`

- [ ] **Step 1: Write the failing API tests**

```python
# tests/test_api.py（替换旧 chat 用例，保留 extract 用例）
import asyncio
import json

from langchain_core.tools import tool
from sqlalchemy import select

from app.config import get_settings
from app.db import get_db_session_factory
from app.llm import get_chat_model
from app.main import app, get_tool_registry
from app.models import Conversation, Message
from app.repositories.conversations import append_message, create_conversation
from app.tools.registry import ToolRegistry
from tests.fakes import FakeToolCallModel, make_settings


def override_model(model):
    app.dependency_overrides[get_chat_model] = lambda: model


def test_chat_creates_conversation_and_returns_meta(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(final_chunks=["你好"]))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "你好"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: meta" in body
    assert '"conversation_id": 1' in body
    assert '"done": true' in body
    with session_factory() as session:
        convo = session.scalar(select(Conversation))
        rows = session.scalars(select(Message).order_by(Message.id)).all()
    assert convo.status == "进行中"
    assert [row.role for row in rows] == ["user", "assistant"]


def test_chat_continues_existing_conversation(client, session_factory):
    with session_factory() as session:
        convo = create_conversation(session, "u1")
        append_message(session, convo.id, role="user", content="第一轮")
        append_message(session, convo.id, role="assistant", content="第一轮回复")

    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    model = FakeToolCallModel(final_chunks=["第二轮回复"])
    override_model(model)
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "conversation_id": convo.id, "message": "第二轮"},
    ) as response:
        body = "".join(response.iter_text())

    assert f'"conversation_id": {convo.id}' in body
    assert any("第一轮" in str(message.content) for message in model.final_messages)
    assert any("第二轮" in str(message.content) for message in model.final_messages)


def test_chat_unknown_conversation_returns_404(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(final_chunks=["不应调用"]))
    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "conversation_id": 99999, "message": "你好"},
    )
    assert response.status_code == 404


def test_chat_rejects_old_messages_payload_and_blank_message(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    old = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    blank = client.post("/api/chat", json={"user_id": "u1", "message": "   "})
    assert old.status_code == 422
    assert blank.status_code == 422


def test_chat_tool_completed_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "query_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["预计明天送达。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: tool" in body
    assert '"status": "running"' in body
    assert '"status": "completed"' in body
    assert '"tool_calls": ["query_logistics"]' in body
    assert "event: error" not in body


@tool
async def slow_logistics(order_id: str) -> str:
    """慢查询，用于超时测试。"""
    await asyncio.sleep(0.2)
    return json.dumps({"ok": True, "data": {"order_id": order_id}})


@tool
def broken_logistics(order_id: str) -> str:
    """总是失败，用于错误帧测试。"""
    raise RuntimeError("boom")


def override_registry(tool_obj):
    registry = ToolRegistry()
    registry.register(tool_obj, label="查物流")
    app.dependency_overrides[get_tool_registry] = lambda: registry


def test_chat_tool_timeout_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    app.dependency_overrides[get_settings] = lambda: make_settings(
        tool_timeout_seconds=0.01,
        tool_retry_delay_seconds=0,
    )
    override_registry(slow_logistics)
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "slow_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["物流暂时查不到，请稍后再试。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert '"status": "retrying"' in body
    assert '"status": "timeout"' in body
    assert '"error_code": "timeout"' in body
    assert "event: error" not in body


def test_chat_tool_failure_frame(client, session_factory):
    app.dependency_overrides[get_db_session_factory] = lambda: session_factory
    override_registry(broken_logistics)
    override_model(FakeToolCallModel(
        tool_calls=[{
            "name": "broken_logistics",
            "args": {"order_id": "1001"},
            "id": "call_1",
            "type": "tool_call",
        }],
        final_chunks=["查询失败，已为您转接人工。"],
    ))
    with client.stream(
        "POST",
        "/api/chat",
        json={"user_id": "u1", "message": "订单 1001 的物流到哪了"},
    ) as response:
        body = "".join(response.iter_text())

    assert '"status": "failed"' in body
    assert '"error_code": "execution_error"' in body
    assert "event: error" not in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`

Expected: FAIL，路由仍读 `request.messages`；`get_db_session_factory` 不存在；新请求体 422/500。

- [ ] **Step 3: Implement DB dependency and route**

- `app/db.py`：新增 `get_db_session_factory() -> sessionmaker[Session]`，返回缓存的 `get_session_factory()`；保留 `get_db_session`。
- `app/main.py`：新增 `get_tool_registry` 依赖；chat 路由在返回 `StreamingResponse` 前用 session 完成：
  - 有 `conversation_id` -> `get_conversation`，`None` 抛 404；
  - 无 id -> `create_conversation(user_id)`；
  - `append_message(role="user", content=request.message)`；
  - 记录 `conversation_id` / `user_id`，关闭 session；
  - 返回 `StreamingResponse(stream_chat(...))`，headers 保持不变。
- 删除 `to_lc_message` 与旧 `messages` 处理；`ChatMessage` 已删除。
- `get_tool_registry` 依赖 `session_factory`，返回 `build_tool_registry(session_factory)`，便于测试覆盖。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py tests/test_static.py -v`

Expected: PASS；`/` 与 `/health` 仍可用，`/api/after-sales/extract` 用例不受影响。

- [ ] **Step 5: Commit**

```bash
git add app/db.py app/main.py tests/test_api.py
git commit -m "feat: wire server-managed chat sessions into API" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: 建表/种子脚本

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/init_db.py`
- Test: `tests/test_init_db.py`

**Interfaces:**
- Consumes: `db/schema.sql`、`app.db.get_engine`、`app.repositories.faq.seed_faq`
- Produces:
  - `read_schema_statements(path: Path) -> list[str]`
  - `table_state(existing: set[str]) -> Literal["empty", "ready", "partial"]`
  - `apply_schema(engine: Engine, path: Path = SCHEMA_PATH) -> str`
  - `main() -> int`

- [ ] **Step 1: Write the failing script tests**

```python
# tests/test_init_db.py
from pathlib import Path

from scripts.init_db import read_schema_statements, table_state

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def test_read_schema_statements_returns_four_create_tables():
    statements = read_schema_statements(SCHEMA_PATH)
    assert len(statements) == 4
    assert statements[0].startswith("CREATE TABLE conversations")
    assert statements[1].startswith("CREATE TABLE messages")
    assert statements[2].startswith("CREATE TABLE faq")
    assert statements[3].startswith("CREATE TABLE tickets")


def test_table_state():
    tables = {"conversations", "messages", "faq", "tickets"}
    assert table_state(set()) == "empty"
    assert table_state(tables) == "ready"
    assert table_state({"conversations"}) == "partial"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_init_db.py -v`

Expected: FAIL，`ModuleNotFoundError: scripts`。

- [ ] **Step 3: Implement the script**

- `read_schema_statements`：读取 UTF-8，去掉 `--` 行注释，按 `;` 切分，strip 后只保留以 `CREATE TABLE` 开头的语句。
- `table_state`：现有集合与四张表比较，返回 `empty` / `ready` / `partial`。
- `apply_schema`：`inspect(engine).get_table_names()`；`empty` -> 逐条 `connection.execute(text(stmt))` 并 commit；`ready` -> 返回 `"skipped"`；`partial` -> 抛 `RuntimeError`，不自动补建/删除。
- `main`：开头 `sys.stdout.reconfigure(encoding="utf-8")`（Windows 控制台默认 GBK，打印中文会 `UnicodeEncodeError`）；`apply_schema(get_engine())`；用 `get_session_factory()()` 调 `seed_faq`；打印表状态与灌入条数；不打印密码；返回 0。

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_init_db.py -v`

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts tests/test_init_db.py
git commit -m "feat: add idempotent schema and FAQ seed script" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: 工具选择评估集与评分脚本

**Files:**
- Create: `evals/tool_calling_cases.jsonl`
- Create: `evals/eval_tool_calling.py`
- Test: `tests/test_tool_eval_scoring.py`

**Interfaces:**
- Consumes: `build_tool_registry`、`build_chat_model`、`TOOL_USAGE_RULES`
- Produces:
  - `score_tool_call(expected_tool: str, expected_args: dict, tool_calls: list[dict]) -> bool`
  - `load_cases(path: Path) -> list[dict]`
  - `main() -> int`

- [ ] **Step 1: Write the failing scoring tests**

```python
# tests/test_tool_eval_scoring.py
from evals.eval_tool_calling import score_tool_call


def test_score_tool_call_accepts_name_and_args_subset():
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_logistics", "args": {"order_id": "1001", "extra": "ok"}}],
    ) is True


def test_score_tool_call_rejects_wrong_name_or_missing_args():
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_order", "args": {"order_id": "1001"}}],
    ) is False
    assert score_tool_call(
        "query_logistics",
        {"order_id": "1001"},
        [{"name": "query_logistics", "args": {}}],
    ) is False
    assert score_tool_call("query_logistics", {"order_id": "1001"}, []) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_eval_scoring.py -v`

Expected: FAIL，`ModuleNotFoundError: evals.eval_tool_calling`。

- [ ] **Step 3: Implement cases and evaluator**

- `evals/tool_calling_cases.jsonl` 至少 10 条，覆盖：
  - `订单 1001 的物流到哪了` -> `query_logistics` / `{"order_id":"1001"}`
  - `帮我查订单 1001` -> `query_order`
  - `商品 P100 还有货吗` -> `query_product`
  - `退货政策是什么` -> `query_faq` / `{"keyword":"退货政策"}`
  - `邮费是多少` -> `query_faq` / `{"keyword":"邮费"}`（known limitation，只验证选择与关键词）
  - `我要人工投诉，订单 1001 一直不发货` -> `create_ticket` / `{"ticket_type":"投诉"}`
  - `帮我转人工处理退货` -> `create_ticket` / `{"ticket_type":"售后"}`
  - `我要人工咨询发票问题` -> `create_ticket` / `{"ticket_type":"咨询"}`
  - 另外 2 条订单/物流变体。
- `eval_tool_calling.py`：`load_cases` 读 JSONL；`main` 开头 `sys.stdout.reconfigure(encoding="utf-8")`；用 `get_settings()` 构造 model；用 `sessionmaker(bind=create_engine("sqlite://"))` 构造 registry（不执行工具、不建表、不写库）；`await model.bind_tools(registry.bindable_tools()).ainvoke([SystemMessage(客服 Prompt + "\n\n" + TOOL_USAGE_RULES), HumanMessage(question)])`；取 `tool_calls[0]` 调 `score_tool_call`；打印 expected/actual/pass；通过率阈值 0.8；未配置 `.env` 返回 2。
- `score_tool_call`：工具名完全匹配，`expected_args` 是实际 args 的子集。

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tool_eval_scoring.py -v`

Expected: PASS。真实模型评估在 Task 12 执行。

- [ ] **Step 5: Commit**

```bash
git add evals/tool_calling_cases.jsonl evals/eval_tool_calling.py tests/test_tool_eval_scoring.py
git commit -m "feat: add tool selection eval set" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: 聊天页 Vibe Coding 改造

**Files:**
- Modify: `app/static/index.html`（CSS 约 59–117；脚本约 141–359）

**Interfaces:**
- Consumes: 8.1 的 `meta` / `tool` / `delta` / `done` / `error` 帧
- Produces: 浏览器端会话 id 保持、工具徽章、新请求体

- [ ] **Step 1: Implement the page changes**

- 请求体改为 `{user_id: "demo-user", conversation_id: conversationId, message: text}`；删除发送完整 `messages` 的逻辑。
- 新增 `conversationId` 变量；`event: meta` 时保存 `payload.conversation_id`。
- 助手气泡改为“徽章区 + 文本区”：`div.bubble.assistant > div.tool-badges + div.bubble-text`；`delta` 只追加到文本区。
- `event: tool` 处理：
  - `running` -> `正在${label}…`
  - `retrying` -> `${label}重试中（${attempt}/${max_attempts}）…`
  - `completed` -> `${label}`
  - `timeout` -> `${label}超时` + `.timeout`
  - `failed` -> `${label}失败` + `.failed`
  - 按 `tool_call_id` 更新同一徽章，不重复创建。
- 清空对话时重置 `conversationId`；收到 404 时清除本地 id 并提示“会话已失效，请重新发送”。
- 保留流式禁用按钮、自动滚动、错误提示、Enter 发送。

- [ ] **Step 2: Run the static test**

Run: `uv run pytest tests/test_static.py -v`

Expected: PASS（页面仍可被 `/` 返回）。

- [ ] **Step 3: Manual Vibe Coding verification**

启动 `uv run uvicorn app.main:app --reload --port 8000`，打开 <http://127.0.0.1:8000/>，确认：

1. 首次提问出现 `meta` 后能继续第二轮；
2. 工具执行时出现“正在查物流…”徽章，成功后变“查物流”；
3. 工具失败/超时有错误样式；
4. 清空后下一轮创建新会话。

该步骤不写前端自动化测试；结果记录到 `dev-notes/ch02.md`。

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html
git commit -m "feat: show tool badges in chat page" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 12: README、实机验收与交付记录

**Files:**
- Modify: `README.md`
- Modify: `dev-notes/ch02.md`

**Interfaces:**
- Consumes: 全部实现
- Produces: 可复现的演示命令、测试/评估结果、验收记录

- [ ] **Step 1: Update README**

- 标题/简介改为 Chapter 02；删除“本章不做工具调用/会话持久化”。
- 目录结构加入 `models.py`、`repositories/`、`tools/`、`scripts/init_db.py`、`db/schema.sql`、`evals/eval_tool_calling.py`。
- 快速开始加入 `docker compose up -d` 与 `uv run python -m scripts.init_db`。
- API 部分改为新请求体；删除无状态 `messages` 示例，加入 `meta` / `tool` / `delta` / `done` 帧示例与三条验收 curl。
- 聊天页说明改为服务端管历史、工具徽章、刷新新会话。
- 测试部分加入 `uv run python -m evals.eval_tool_calling`。
- 已知限制加入 `query_faq` 字面 LIKE 漏召回、工具结果持久化截断、单轮工具调用。

- [ ] **Step 2: Run all tests**

Run: `uv run pytest -q`

Expected: 全部通过（数量以实际为准，不少于 60）。

- [ ] **Step 3: Run MySQL init and verify tables/seed**

Run:

```bash
docker compose up -d
uv run python -m scripts.init_db
```

Expected: 四张表存在；`faq` 首次灌入 10 条，重复执行返回 0；不打印密码。

Verify:

```bash
docker compose exec -T mysql sh -c 'mysql -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" customer_service -e "SELECT COUNT(*) AS faq_count FROM faq; SHOW TABLES;"'
```

Expected: `faq_count` 为 10；四张表齐全。

- [ ] **Step 4: Run real-model eval and acceptance**

Run:

```bash
uv run python -m evals.eval_tool_calling
```

Expected: 打印逐条选择结果与通过率；阈值 0.8。若模型不支持工具调用，停止并询问，不换模型。

Start `uv run uvicorn app.main:app --reload --port 8000`，执行三条验收：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"订单 1001 的物流到哪了"}'

curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"退货政策是什么"}'

curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"邮费是多少"}'
```

Expected:

1. 第一条出现 `tool` 的 `running` / `completed`，最终回答基于模拟物流数据；
2. 第二条选中 `query_faq`，能答出退货政策；
3. 第三条选中 `query_faq`，关键词 `邮费` 未命中，助手如实说明并建议转人工；记录为 known limitation，不作为 bug 修复。

浏览器打开聊天页，重复以上三条，确认工具徽章与流式回答；结果记录到 `dev-notes/ch02.md`。

- [ ] **Step 5: Append dev-notes and commit**

在 `dev-notes/ch02.md` 追加 Task 1–12 的阶段记录（每条四项：用户关键原话、关键产出、用户拒绝或纠偏、翻车与返工），至少包含：

- 设计批准与三条口径；
- 模型/仓储/工具/编排/API 的实现与测试结果；
- 工具失败/超时帧与截断落库的验证；
- MySQL 建表/种子/真实请求结果；
- `query_faq` 漏召回 known limitation 的实测记录；
- 浏览器验收结果与评估通过率。

```bash
git add README.md dev-notes/ch02.md
git commit -m "docs: record chapter 2 delivery and acceptance" -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** 数据模型 Task 1；仓储/种子 Task 2；请求/配置 Task 3；工具注册/执行/超时/重试 Task 4；FAQ/工单 Task 5；Prompt Task 6；编排/落库/截断/工具对修复 Task 7；API Task 8；建表脚本 Task 9；评估 Task 10；聊天页 Task 11；验收 Task 12。
- **Type consistency:** `ToolRegistry.execute_stream` 的事件/结果类型在 Task 4 定义，Task 7 消费；`stream_chat` 签名在 Task 7 定义，Task 8 调用；`ChatRequest` 在 Task 3 定义，Task 8 使用。
- **Review Focus tests:** 工具对修复 Task 7；超时/重试终点 Task 4/8；截断 Task 7；`create_ticket` 不可重试与状态 Task 4/5；FAQ wildcard/known limitation Task 2/5。
- **Known limitation:** `query_faq` 的“邮费”漏召回在 spec 15.1、Task 2 种子、Task 5 miss、Task 10 评估、Task 12 验收中均明确为预期行为。
