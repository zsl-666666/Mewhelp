# Chapter 02 设计：客服聊天接入 Function Calling 工具链

- 日期：2026-11-19
- 状态：已批准（2026-11-19；含用户补充的三条口径）
- 分支：chapter-01（当前工作分支）
- 关联留痕：`dev-notes/ch02.md`
- 基线：Chapter 01 纯对话实现与设计

## 1. 背景与目标

Chapter 01 已经跑通无状态 SSE 纯对话。本章要把“查数据”的能力长进客服聊天本身：模型自己决定调用哪个业务工具，后端执行一次工具、把结果回灌给模型，最终回答继续逐 token 流式输出，并把整轮对话（含工具调用与结果）落到 MySQL。

本章要证明五件事：

1. FastAPI + SQLAlchemy 2.x 的分层骨架可以支撑正式的业务表与查询。
2. 五个 LangChain `@tool` 工具可以被注册、做参数校验、错误处理、超时重试。
3. SSE 链路能在最终回答前推送工具状态帧，前端能在气泡里显示工具徽章。
4. 聊天记录能按 `conversations` / `messages` 持久化，工具调用与结果可追溯。
5. 只做**单轮工具调用**：模型选中一个工具、执行、收敛成最终回答，不做 Agent Loop。

## 2. 验收标准

1. 浏览器打开聊天页，问“订单 1001 的物流到哪了”：
   - 模型选中 `query_logistics`；
   - 气泡里出现“查物流”徽章；
   - 最终回答基于工具返回的模拟物流数据。
2. 问“退货政策是什么”：
   - 模型选中 `query_faq`；
   - 能查到 `faq` 表中的退货政策并作答。
3. 问“邮费是多少”：
   - 模型选中 `query_faq`；
   - 关键词 `邮费` 在种子数据中查不到（种子数据只有“运费”）；
   - 助手如实说明没有查到，并给出转人工等建议。
   - **这个漏召回是 `query_faq` 字面 LIKE 检索的 known limitation，不是 bug，也不是本章要修的缺陷。** 必须记录在 `dev-notes/ch02.md`，留给下一步（同义词/向量检索）升级；本章不得通过给种子数据补“邮费”或让模型改写关键词来掩盖。
4. `create_ticket` 能在用户要求人工处理时写入 `tickets` 表，并把会话状态改为“已转人工”。
5. 离线单元测试通过；真实模型的工具选择评估集可运行，结果记录在 dev-notes。

## 3. 非目标

- 多轮自动循环的 Agent Loop、LangGraph。
- 向量检索、RAG、FAQ 语义召回。
- 真实电商/物流 API 接入；订单、商品、物流只返回模拟数据，不建表。
- 用户注册、登录、鉴权、多租户。
- 生产级迁移工具（Alembic）；本章用 `db/schema.sql` + 幂等种子脚本。
- 聊天页工程化重构；聊天页继续走 Vibe Coding 例外流程。
- 刷新页面后恢复历史会话；本章只保证同一页面会话内连续对话。

## 4. 约束

- Python 3.13、uv、FastAPI、SQLAlchemy 2.x、PyMySQL、MySQL 8.0（Docker）。
- 模型统一走 OpenAI 兼容协议，当前 `.env` 使用 DeepSeek。
- 数据库字符集 `utf8mb4`，排序规则 `utf8mb4_unicode_ci`。
- 真实密码只在本机 `.env`，不得进入代码、Compose、测试、日志或 Git。
- 涉及具体库/API 的写法，动手前用 Context7 核对；Context7 会话内工具未挂载时，沿用 HTTP 端点 `https://mcp.context7.com/mcp`。
- 技术选型定死；真实模型不支持工具调用时停下询问，不擅自换模型/框架。

## 5. 方案选择：会话与历史归属

现状 `/api/chat` 是无状态的，客户端每轮带完整 `messages`，服务端不保存会话。新要求要落 `conversations` / `messages`，因此必须先决定会话标识与历史归属。

| 方案 | 做法 | 取舍 |
|---|---|---|
| A. 保留无状态请求 + 可选 conversation_id | 客户端仍带完整 `messages`；服务端只额外持久化本轮消息 | 改动最小；但“历史”仍由客户端掌握，落库更像旁路记录，客户端与服务端可能不一致 |
| **B. 服务端管历史（选中）** | 前端只发 `conversation_id` + 新 `message`；后端从 `messages` 表加载上下文 | 会话真正落库、历史唯一来源、后续加鉴权/多轮更自然；接口与前端需改造，ch01 的无状态契约被替换 |
| C. 每次请求新建会话 | 每轮都建 conversation | 能写表，但 `conversations` 失去“会话”意义，不采用 |

**选择 B 的理由**：用户要求“按照新的要求改”，并且本章明确要 `conversations` 会话壳与 `messages` 消息流水。服务端管历史能让工具调用、ToolMessage、最终回答按同一会话串起来，也避免客户端历史与服务端记录分叉。代价是 `/api/chat` 请求体变化、ch01 相关测试与 README 需要同步更新，这是有意为之的破坏性变更。

### 5.1 已定的实现取舍

- `user_id` 为演示字符串，默认 `"demo-user"`；本章不做用户表与鉴权。
- 首次请求不带 `conversation_id` 时，后端创建会话并通过 SSE `meta` 帧返回 id；后续请求带上它。
- 未知 `conversation_id` 返回 404，不静默新建，避免客户端 bug 被掩盖。
- 聊天页只在当前页面生命周期内保存 `conversation_id`；刷新后开启新会话，旧会话仍留在数据库。
- 数据库表以用户提供的 `db/schema.sql` 为唯一事实来源，由 `scripts/init_db.py` 执行 DDL 创建；ORM 模型只做映射，不用 `create_all` 建 MySQL 表。种子脚本幂等，只灌 FAQ。`conversations` / `messages` / `tickets` 由运行时写入。
- 单测使用 SQLite 内存库；MySQL 实机用种子脚本 + 真实请求验证。

## 6. 架构与目录

```text
app/
  main.py                    FastAPI 路由（/、/health、/api/chat）
  config.py                  主配置（LLM + 工具超时/重试）
  db.py                      现有 Engine / Session 基础设施
  models.py                  Base + Faq / Conversation / Message / Ticket
  schemas.py                 ChatRequest 等请求模型
  sse.py                     SSE 帧格式化
  repositories/
    __init__.py
    conversations.py         会话创建、历史加载、消息追加
    faq.py                   FAQ LIKE 查询
    tickets.py               工单写入
  services/
    chat.py                  聊天编排：加载历史、工具决策、执行、回灌、流式回答、落库
    context.py               token 预算裁剪 + 工具调用/结果配对修复
  tools/
    __init__.py              build_tool_registry(session_factory)
    registry.py              ToolRegistry / ToolExecutionResult / 工具标签
    ecommerce.py             query_order / query_product / query_logistics
    knowledge.py             query_faq
    tickets.py               create_ticket
  static/index.html          聊天页（Vibe Coding）
db/
  schema.sql                 权威建表 DDL（用户提供）
scripts/
  init_db.py                 执行 DDL + 幂等灌 FAQ
evals/
  tool_calling_cases.jsonl   工具选择标注样例
  eval_tool_calling.py       真实模型工具选择评估
tests/
  test_models.py
  test_seed.py
  test_tools.py
  test_registry.py
  test_chat.py               改写
  test_api.py                改写
  test_schemas.py            补充新请求模型
docs/superpowers/specs/2026-11-19-ch02-function-calling-design.md
dev-notes/ch02.md
README.md                    更新运行、演示、验收说明
```

分层原则：

- 路由只做参数校验、会话准备、响应封装。
- `services/chat.py` 编排 LangChain 与 SSE；不直接写 SQL。
- `repositories/` 封装 SQLAlchemy 查询与写入。
- `tools/` 只关心工具契约与业务返回，不直接拼 SQL。
- `models.py` 只放 ORM 模型；`db.py` 只放连接基础设施。

## 7. 数据模型

**权威来源**：`db/schema.sql`，内容为用户 2026-11-19 提供的 DDL。MySQL 表由该 DDL 创建，ORM 模型只负责映射，不使用 `create_all` 建 MySQL 表；SQLite 单测用模型的可移植类型建内存表。

所有表统一 `ENGINE=InnoDB`、`DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci`。DDL 原文只写 `DEFAULT CHARSET=utf8mb4`，MySQL 8.0 会落成 `utf8mb4_0900_ai_ci`；`db/schema.sql` 已显式补上 `COLLATE=utf8mb4_unicode_ci`，与项目要求一致。

### 7.1 `faq`

| 列 | 类型 | 约束 |
|---|---|---|
| `id` | BIGINT UNSIGNED | PK, autoincrement |
| `question` | VARCHAR(512) | NOT NULL |
| `answer` | TEXT | NOT NULL |
| `category` | VARCHAR(64) | NOT NULL, index `idx_category` |
| `created_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP |
| `updated_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP, ON UPDATE CURRENT_TIMESTAMP |

`question` 没有唯一约束；种子脚本的幂等靠“表为空才灌”，不靠唯一键。

### 7.2 `conversations`

| 列 | 类型 | 约束 |
|---|---|---|
| `id` | BIGINT UNSIGNED | PK, autoincrement |
| `user_id` | VARCHAR(64) | NOT NULL, index `idx_user_id` |
| `status` | ENUM('进行中','已转人工','已结束') | NOT NULL, default '进行中' |
| `created_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP |
| `updated_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP, ON UPDATE CURRENT_TIMESTAMP |

新建会话 `status='进行中'`；`create_ticket` 成功后置为 `'已转人工'`。本章没有关闭会话入口，`'已结束'` 暂不使用。

### 7.3 `messages`

| 列 | 类型 | 约束 |
|---|---|---|
| `id` | BIGINT UNSIGNED | PK, autoincrement |
| `conversation_id` | BIGINT UNSIGNED | NOT NULL, FK `conversations.id`, index `idx_conversation_id` |
| `role` | ENUM('user','assistant','tool') | NOT NULL |
| `content` | TEXT | NULL；assistant 纯工具调用时可为空 |
| `tool_calls` | JSON | NULL；assistant 的工具调用申请 |
| `tool_call_id` | VARCHAR(64) | NULL；tool 消息对应的调用 id |
| `created_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP |

外键没有 `ON DELETE CASCADE`；本章不提供删除会话接口。

### 7.4 `tickets`

| 列 | 类型 | 约束 |
|---|---|---|
| `ticket_no` | VARCHAR(32) | PK |
| `conversation_id` | BIGINT UNSIGNED | NOT NULL, FK `conversations.id`, index `idx_conversation_id` |
| `description` | TEXT | NOT NULL |
| `ticket_type` | ENUM('售后','投诉','咨询') | NOT NULL |
| `status` | ENUM('待处理','已处理') | NOT NULL, default '待处理' |
| `created_at` | DATETIME | NOT NULL, default CURRENT_TIMESTAMP |

`create_ticket` 只创建“待处理”工单，并在同一事务内把所属 `conversations.status` 改为 `'已转人工'`；本章不提供把工单改为“已处理”的接口。

关系：`Conversation.messages` 一对多，`Conversation.tickets` 一对多；无级联删除。ORM 模型用可移植类型映射 MySQL 的 ENUM 与 BIGINT UNSIGNED，SQLite 单测用 `String` + `CheckConstraint` / `BigInteger.with_variant(Integer, "sqlite")` 建表。

### 7.5 消息落库顺序

一轮带工具的对话写四条消息：

1. `user`：用户问题。
2. `assistant`：工具调用申请，`tool_calls` 为 JSON，`content` 保存模型调用工具前的引导语；没有引导语时写 `NULL`。
3. `tool`：工具结果，`tool_call_id` 对应第 2 条的调用 id。
4. `assistant`：最终回答。

无工具的对话只写第 1、4 条。

### 7.6 `tool` 角色落库字段约定（批准补充口径 ①）

`tool` 行必须自描述，便于审计与历史重建。字段约定：

| 字段 | 值 |
|---|---|
| `role` | `"tool"` |
| `conversation_id` | 当前会话 id |
| `tool_call_id` | assistant `tool_calls[0].id`，必填 |
| `tool_calls` | `NULL` |
| `content` | 下述 JSON envelope 的字符串 |
| `created_at` | 数据库默认 |

`content` envelope（`ensure_ascii=False`、紧凑分隔符）：

```json
{
  "tool": "query_logistics",
  "arguments": {"order_id": "1001"},
  "ok": true,
  "result": "{\"ok\":true,\"data\":{\"order_id\":\"1001\",\"status\":\"运输中\"}}",
  "truncated": false,
  "original_chars": 58,
  "attempts": 1,
  "duration_ms": 12
}
```

失败/超时：

```json
{
  "tool": "query_logistics",
  "arguments": {"order_id": "1001"},
  "ok": false,
  "result": "{\"ok\":false,\"error\":\"工具执行超时\"}",
  "error_code": "timeout",
  "retryable": true,
  "truncated": false,
  "original_chars": 39,
  "attempts": 3,
  "duration_ms": 5003
}
```

约定：

- `tool`：工具名。
- `arguments`：模型实际给出的参数，**不包含**服务端注入的 `conversation_id`；注入关系由 `messages.conversation_id` 体现。
- `result`：模型实际收到的工具结果 JSON 字符串（通常是 `{"ok": true, "data": ...}` 或 `{"ok": false, "error": ...}`）。
- **截断**：`result` 超过 `TOOL_RESULT_MAX_CHARS = 4000` 字符时，只持久化前 4000 字符，`truncated=true`，`original_chars` 记录原始长度；当前轮模型仍收到完整结果，后续从历史重建时只能看到截断版本。这是有意的存储保护，写入 known limitations。
- `error_code`：`timeout` / `validation_error` / `execution_error` / `unknown_tool` / `internal_error`。
- `retryable`：错误类别是否可重试；最终帧的 `attempts` 表示实际尝试次数。
- `attempts` / `duration_ms`：本次工具执行的实际尝试次数与总耗时。
- 缺少必填字段时按 `internal_error` 处理，不写半结构化 content。

assistant 工具调用行对应约定：

| 字段 | 值 |
|---|---|
| `role` | `"assistant"` |
| `content` | 工具调用前的引导语；没有则为 `NULL` |
| `tool_calls` | 仅第一个调用的 JSON 数组，元素形如 `{"name":"query_logistics","args":{"order_id":"1001"},"id":"call_abc","type":"tool_call"}` |
| `tool_call_id` | `NULL` |

历史重建时：`assistant` 行用 `AIMessage(content, tool_calls=tool_calls)`；`tool` 行用 `ToolMessage(content=payload["result"], tool_call_id=tool_call_id)`。若 `truncated=true`，`payload["result"]` 就是截断后的字符串，直接作为 ToolMessage content。

## 8. API 契约

### 8.1 `POST /api/chat`（SSE）

请求体（破坏性变更）：

```json
{
  "user_id": "demo-user",
  "conversation_id": 12,
  "message": "订单 1001 的物流到哪了"
}
```

- `user_id`：可选，默认 `"demo-user"`，1–64 字符。
- `conversation_id`：可选，正整数；不存在返回 404。
- `message`：必填，非空白。

SSE 帧顺序（正常）：

```text
event: meta
data: {"conversation_id": 12, "user_id": "demo-user"}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "running", "arguments": {"order_id": "1001"}}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "completed", "ok": true, "attempts": 1, "duration_ms": 12}

data: {"delta": "您的订单"}

data: {"delta": "预计明天送达。"}

data: {"done": true, "conversation_id": 12, "trimmed": false, "tool_calls": ["query_logistics"]}
```

工具重试/超时/失败时必须补帧（批准补充口径 ②）：

```text
event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "retrying", "attempt": 2, "max_attempts": 3, "error_code": "timeout", "error": "工具执行超时"}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "timeout", "ok": false, "error_code": "timeout", "error": "工具执行超时", "retryable": true, "attempts": 3, "duration_ms": 5003}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "failed", "ok": false, "error_code": "validation_error", "error": "参数校验失败: order_id 不能为空", "retryable": false, "attempts": 0, "duration_ms": 0}
```

- `status` 取值为 `running` / `retrying` / `completed` / `timeout` / `failed`。
- `running`：工具开始执行，带 `arguments`。
- `retrying`：一次可重试失败后、下一次尝试前发出；带 `attempt`（即将进行的第几次）、`max_attempts`、`error_code`、`error`。不可重试错误不发 `retrying`。
- `completed`：`ok=true`，带 `attempts`、`duration_ms`。
- `timeout`：**超时专用最终帧**，`ok=false`，`error_code="timeout"`，带 `retryable`、`attempts`、`duration_ms`。
- `failed`：非超时最终失败帧，`error_code` 为 `validation_error` / `execution_error` / `unknown_tool` / `internal_error`，带 `retryable`、`attempts`、`duration_ms`。
- 每个工具调用必须以 `completed` / `timeout` / `failed` 之一收尾，不能只发 `running`。
- 无论最终是 `completed` 还是 `timeout` / `failed`，模型都会收到对应的工具结果（成功结果或错误 JSON），然后继续生成最终回答；不直接中断 SSE。
- 无工具时：`meta` → 若干 `delta` → `done`。
- 模型/上游失败：沿用 `event: error` + `data: {"code":"upstream_error","message":"..."}`；工具失败本身不走 `event: error`。
- `done` 帧保留原有 `done` / `trimmed` 字段，新增 `conversation_id` 与 `tool_calls`（工具名数组，失败/超时也记录）。

### 8.2 会话准备

路由在返回 `StreamingResponse` 之前同步完成：

1. 校验请求。
2. 有 `conversation_id`：查询会话，不存在抛 404。
3. 无 `conversation_id`：创建 `Conversation(user_id, status="进行中")`。
4. 插入本轮 `user` 消息并提交。

这样即使后续模型失败，用户消息也已经落库；`meta` 帧也能立即返回新会话 id。

### 8.3 错误码

| 场景 | 行为 |
|---|---|
| `message` 空白 | 422 |
| `conversation_id` 不存在 | 404 |
| 模型调用失败 | HTTP 200 + `event: error` |
| 工具参数校验失败 | 不重试；回灌工具错误结果，模型继续回答 |
| 工具超时/执行失败 | 按可重试策略重试；耗尽后回灌错误结果 |
| 数据库写入失败 | 记录日志；已发出的 SSE 帧无法撤回，最终以 `event: error` 收尾 |

## 9. 工具设计

### 9.1 五个工具

| 工具 | 参数 | 数据来源 | 重试 |
|---|---|---|---|
| `query_order(order_id)` | 订单号 | 按订单号做确定性伪随机模拟 | 可重试 |
| `query_product(product_id)` | 商品号 | 按商品号做确定性伪随机模拟 | 可重试 |
| `query_logistics(order_id)` | 订单号 | 按订单号做确定性伪随机模拟 | 可重试 |
| `query_faq(keyword, limit=3)` | 关键词、条数 | `faq` 表 SQL LIKE | 可重试 |
| `create_ticket(description, ticket_type, conversation_id)` | 描述、类型、会话 id | 写 `tickets` 表 | **不可重试** |

- 订单/商品/物流不建表；用 `random.Random(f"order:{order_id}")` 之类的确定性种子，保证同一 id 多次查询结果稳定，便于演示与排查。
- `query_faq` 用 `Faq.question.contains(keyword, autoescape=True)` 或 `Faq.answer.contains(...)`；`autoescape` 转义 `%` / `_`。
- `create_ticket` 的 `conversation_id` 用 `Annotated[int, InjectedToolArg]` 注入，不出现在模型可见的参数 Schema 中。`ticket_no` 形如 `T{YYYYMMDDHHMMSS}{4位随机HEX}`，总长 19，符合 `VARCHAR(32)`。
- `ticket_type` 只能取 `售后 / 投诉 / 咨询`，与 `tickets.ticket_type` 的 ENUM 一致；工具 docstring 要求模型按语义选择，例如退货/退款/物流问题归“售后”，不满/服务态度归“投诉”，一般咨询归“咨询”。
- `create_ticket` 成功后，在同一事务内把 `conversations.status` 更新为 `'已转人工'`；失败或参数校验失败则不建单、不改状态。

### 9.2 注册与执行

`ToolRegistry` 负责：

- 注册 `BaseTool`、按名称查找、提供绑定给模型的工具列表。
- 参数 Schema 校验：执行前用 `tool.args_schema.model_validate(args)`；校验失败直接返回错误，不重试。
- 注入参数：执行时把 `conversation_id` 等注入参数与模型参数合并后传给 `tool.ainvoke`。
- 超时：`asyncio.wait_for(..., timeout=settings.tool_timeout_seconds)`，默认 5 秒。
- 重试：可重试工具最多重试 `settings.tool_max_retries` 次（默认 2），每次间隔 `settings.tool_retry_delay_seconds`（默认 0.2 秒）。不可重试工具只执行一次。
- 执行事件流：`execute_stream(tool_call, ...)` 逐事件产出 `started` / `retrying` / `completed` / `timeout` / `failed`，最终事件携带 `ToolExecutionResult`；SSE 层直接消费该异步生成器转成 `tool` 帧，单测消费同一事件流断言重试次数与错误分类。
- 错误处理：捕获异常，记录完整堆栈；返回给模型的 `content` 是脱敏 JSON，如 `{"ok": false, "error": "工具执行超时"}`。
- 结果结构：`ToolExecutionResult(ok, content, error, error_code, retryable, attempts, duration_ms)`。
- 错误分类：

| `error_code` | 触发场景 | 默认可重试 |
|---|---|---|
| `validation_error` | 参数 Schema 校验失败 | 否 |
| `unknown_tool` | 工具名未注册 | 否 |
| `timeout` | `asyncio.wait_for` 超时 | 是 |
| `execution_error` | 工具内部异常 | 是 |
| `internal_error` | 事件/结果归集失败等兜底 | 否 |

工具自身标记 `retryable=False`（如 `create_ticket`）时，任何错误都不重试，`retryable` 最终为 `false`。

`create_ticket` 设为不可重试，原因是重试可能产生重复工单；这是本章的明确取舍。

### 9.3 工具结果格式

工具返回 JSON 字符串（`ensure_ascii=False`）：

```json
{"ok": true, "data": {...}}
```

FAQ 未命中：

```json
{"ok": true, "data": [], "message": "未找到相关 FAQ"}
```

错误：

```json
{"ok": false, "error": "订单查询超时"}
```

## 10. 聊天编排与 SSE 流程

`services/chat.py` 的核心流程：

1. 通过 `repositories/conversations.py` 从 `messages` 表加载该会话历史，按 `id` 升序重建：
   - `user` → `HumanMessage`
   - `assistant` → `AIMessage(content, tool_calls)`
   - `tool` → `ToolMessage(content, tool_call_id)`
2. 用现有 `trim_history` 按 token 预算裁剪；此时最后一条是本轮用户消息。
3. 拼 System Prompt（追加工具使用规则）。
4. 推 `meta` 帧。
5. 第一次模型调用：`model.bind_tools(registry.tools).astream(messages)`。
   - 逐 chunk 累积为完整 `AIMessageChunk`；
   - 有文本就实时推 `delta`；
   - 流结束后检查 `tool_calls`。
6. 没有工具调用：落库最终 assistant 消息，推 `done`。
7. 有工具调用：
   - 只取第一个 `tool_call`；多于一个时记录警告并丢弃其余，保证单轮收敛。
   - 落库 assistant 工具调用消息（`content` 为引导语；没有引导语时写 `NULL`；`tool_calls` 为第一个调用）。
   - 推 `tool` 状态帧（running）。
   - 消费 `execute_stream`：每次重试推 `retrying` 帧；最终推 `completed` / `timeout` / `failed` 帧。
   - 按 7.6 的 envelope 落库 `tool` 消息（`tool_call_id`、工具名、模型参数、截断后的结果、错误分类、尝试次数与耗时）。
   - 若工具是 `create_ticket` 且成功，`repositories/tickets.py` 已在同一事务内把会话状态改为 `'已转人工'`。
   - 把 `AIMessage(tool_calls=[...])` 与 `ToolMessage` 追加到消息列表。
   - **第二次模型调用使用未绑定工具的普通 model**，强制收敛；逐 token 推 `delta`。
   - 落库最终 assistant 消息，推 `done`。
8. 任何阶段异常：记录日志；尽可能落库已产生的内容；推 `event: error`。

关键约束：

- 第二次模型调用**不再绑定工具**，从机制上保证“只调一次工具就收敛”。
- 第二次调用**不重新裁剪**历史；沿用第一次裁剪后的列表，避免 `trim_history` 丢掉末尾的 `ToolMessage`。
- 第一次裁剪后调用 `repair_tool_pairs`：删除孤立的 `ToolMessage`，以及缺少对应工具结果的 `AIMessage(tool_calls=...)`，避免 OpenAI 兼容接口因工具调用/结果不成对返回 400。修复后的列表用于两次模型调用，保证前后一致。
- 第一次调用如果产生了引导语，会先流式显示，再显示工具徽章，最后追加最终回答；这是可接受的交互。
- 历史裁剪后如果只剩本轮用户消息，也正常继续。

## 11. 前端改造（Vibe Coding 例外）

只改 `app/static/index.html`，不套 brainstorm / TDD / code review。

- 请求体改为 `{user_id, conversation_id, message}`；不再发送完整 `messages`。
- 首次 `meta` 帧保存 `conversation_id`；后续请求带上。
- 新增 `tool` 事件处理：
  - 在助手气泡内创建徽章容器；
  - `status: running` 显示“正在查物流…”；
  - `status: retrying` 显示“查物流重试中（2/3）…”；
  - `status: completed` 显示“查物流”；
  - `status: timeout` 显示“查物流超时”并使用错误样式；
  - `status: failed` 显示“查物流失败”并使用错误样式。
- 助手气泡结构改为“徽章区 + 文本区”；`delta` 追加到文本区。
- 保留现有错误、清空、流式中禁用按钮、自动滚动行为。
- 清空对话时同时清空 `conversation_id`，下一轮会创建新会话；不删除数据库中的旧会话。
- 收到 404（会话不存在）时清除本地 id 并提示用户重发。
- 工具中文标签由后端 `tool` 帧提供，前端不重复维护映射。

## 12. 建表与种子数据

`db/schema.sql` 是权威 DDL。`scripts/init_db.py`：

1. 检查四张表是否存在：
   - 四张都不存在：读取 `db/schema.sql`，逐条执行 `CREATE TABLE`。
   - 四张都存在：跳过建表。
   - 只存在一部分：报错退出，不自动补建/删除，避免掩盖半初始化状态。
2. 若 `faq` 表为空，插入 10 条左右常见问题；若已有数据则跳过。
3. 种子数据必须包含：
   - “退货政策是什么？”及退货政策答案（验收 2）。
   - “运费怎么计算？”等物流/运费问题。
   - **任何问题、答案、分类中都不出现“邮费”二字**，保证验收 3 的预期漏召回。
4. 不灌 `conversations` / `messages` / `tickets` 测试数据；这三张表由运行时产生。
5. 打印建表与种子结果，不打印数据库密码。
6. 不在脚本里用 `Base.metadata.create_all` 建 MySQL 表；ORM 模型只用于查询/写入与 SQLite 单测。

## 13. 测试与评估

### 13.1 后端 TDD

按任务先写失败测试，再实现：

- `test_models.py`：SQLite 内存库建表、JSON 字段、角色/工单类型/会话状态 CHECK 约束、关系、`messages.content` 可空。
- `test_schema.py`：`db/schema.sql` 含四张表的 `CREATE TABLE`，建表顺序为 conversations → messages / faq → tickets，且每张表都含 `COLLATE=utf8mb4_unicode_ci`。
- `test_seed.py`：首次灌入、重复执行幂等、“退货政策”存在、“邮费”不存在（known limitation，不是失败）。
- `test_tools.py`：FAQ 命中/未命中、LIKE 转义、工单写入、随机工具确定性与 JSON 格式、非法参数、工单类型只接受 售后/投诉/咨询、建单后会话状态为“已转人工”。
- `test_registry.py`：未知工具、参数校验失败不重试、超时可重试、耗尽后返回 `timeout`、不可重试工具只执行一次、注入参数不出现在 Schema、`execute_stream` 事件顺序为 started →（retrying）→ completed/timeout/failed、错误分类与 `retryable` 正确。
- `test_chat.py`：无工具路径、单工具路径、多工具只取第一个、工具失败/超时仍收敛、引导语/最终回答流式、落库四条消息、tool 行 envelope 含 `tool_call_id` / `tool` / `arguments` / 截断后的 `result` / `truncated` / `attempts` / `duration_ms`。
- `test_api.py`：新请求体、首次建会话并返回 meta、带 id 续聊、未知 id 404、空白 message 422、SSE 帧契约（running → retrying → timeout/failed/completed 均覆盖；工具失败不误发 `event: error`）。
- `test_context.py`：新增 `repair_tool_pairs` 用例，覆盖裁剪切断工具对后不留下孤立 `ToolMessage` / `AIMessage`。
- 现有 `test_sse.py`、`test_extract.py`、`test_config.py` 保持通过；因请求体变化需要同步调整的 `test_chat.py`、`test_api.py`、`test_schemas.py` 已列入上面的改造。

### 13.2 非单测任务的验证（评估层已定）

- Prompt 与工具选择属于“非可单测代码”的产出，用标注评估集验证。
- **评估只做模型选择层，不执行工具、不写数据库**：
  - `evals/tool_calling_cases.jsonl` 每条包含 `question`、`expected_tool`、`expected_args`（子集匹配）。
  - `evals/eval_tool_calling.py` 只做一次 `bind_tools(...).ainvoke(...)`，比较模型返回的 `tool_calls[0].name` 与关键参数。
  - `create_ticket` 案例只断言选中工具与参数，不真正建工单，避免评估污染 `tickets` 表。
  - FAQ 命中/漏召回由“选中的工具 + keyword 参数”判定；实际检索结果由 `test_seed.py` / `test_tools.py` 对种子数据做确定性验证。评估不需要数据库 fixture。
  - “邮费是多少”只断言模型选中 `query_faq` 且 `keyword` 为“邮费”；字面 LIKE 查不到“运费”是 **known limitation**，不计为评估失败。
- 质量目标：工具选择通过率 ≥ 0.8；FAQ 漏召回按 known limitation 记录。阈值是质量目标，不是交付硬门槛。
- 聊天页走 Vibe Coding：不写自动化前端测试，按验收标准手动浏览器验证，结果记录 dev-notes。

### 13.3 实机验证

- `docker compose up -d` → healthy。
- `uv run python -m scripts.init_db` → 执行 `db/schema.sql` + 种子。
- `uv run pytest` → 全绿。
- 启动 `uv run uvicorn app.main:app --reload --port 8000`。
- `curl -N` 演示三条验收问题，确认 `meta` / `tool` / `delta` / `done` 帧与数据库落库。
- 浏览器打开聊天页，确认工具徽章与流式回答。
- `uv run python -m evals.eval_tool_calling` → 记录通过率。

## 14. Context7 核对结论（2026-11-19）

会话内 Context7 MCP 工具未挂载；`claude mcp list` 显示 context7 HTTP 端点已连接。沿用 ch01 的 HTTP 端点核对。

- `@tool` 装饰器、`model.bind_tools(tools)`、`AIMessage.tool_calls` 是 LangChain 官方工具调用写法。
- `ToolMessage` 需要 `tool_call_id` 与工具结果内容。
- `InjectedToolArg` 位于 `langchain_core.tools`，可把 `conversation_id` 等参数从模型可见 Schema 中隐藏，调用时显式传入。
- 流式工具调用：`AIMessageChunk.tool_call_chunks` 按块拼接；用 `chunk + chunk` 聚合成完整 `AIMessageChunk` 后读取 `tool_calls`。
- 官方文档示例使用 `from langchain.tools import tool`，但本项目未安装 `langchain` 元包；已在本机验证 `from langchain_core.tools import tool, InjectedToolArg` 可用。
- SQLAlchemy 2.x：`DeclarativeBase` + `Mapped` + `mapped_column`；`sessionmaker(engine)` 绑定 Engine；`JSON` 类型可用于消息工具调用字段。本章 MySQL 建表以 `db/schema.sql` 为准，`create_all` 只用于 SQLite 单测。
- `AIMessageChunk` 支持 `__add__`，`ToolMessage` 字段包含 `tool_call_id` / `content`；已在本机 `langchain-core 1.8.1` 验证。

## 15. 风险与待确认

1. **真实模型工具调用能力**：当前 `.env` 使用 DeepSeek。若 `deepseek-chat` 的工具调用/流式工具块不兼容，暂停并询问，不擅自换模型。
2. **known limitation：“邮费”字面漏召回**：`query_faq` 只做字面 LIKE，不保证同义词命中；若模型把“邮费”改写为“运费”会意外命中，验收 3 的预期漏召回失效。System Prompt 明确要求使用用户原话关键词；评估集会验证，必要时调整 Prompt。该漏召回不是 bug，见 15.1。
3. **单轮工具调用的强制执行**：模型可能返回多个工具调用；实现只取第一个并丢弃其余，需在日志与 dev-notes 中记录该行为。
4. **工单重试的重复风险**：`create_ticket` 不可重试；超时发生在提交之后仍可能留下工单，本章接受该边界并记录。
5. **同步 DB + 异步 SSE**：工具通过 `tool.ainvoke` 在线程池执行；流式生成器内的落库用 `asyncio.to_thread`，避免阻塞事件循环。
6. **SQLite 单测与 MySQL 差异**：MySQL 侧由 `db/schema.sql` 建 `ENUM` / `BIGINT UNSIGNED`；ORM 用 `String` + `CheckConstraint`、`BigInteger.with_variant(Integer, "sqlite")` 映射。SQLite 只能验证逻辑与约束，不能替代 MySQL 实机校验。
7. **API 破坏性变更**：ch01 的 `messages` 请求体被替换；README、测试、前端同步更新。这是“按照新的要求改”的直接结果。
8. **刷新恢复历史**：本章不做；刷新后新会话，旧数据仍在库中。若需要，可后续加 `GET /api/conversations/{id}/messages`。
9. **DDL 与 ORM 漂移**：`db/schema.sql` 是权威结构，模型手写映射；若后续改 DDL 未同步模型，会出现运行时错误。`test_schema.py` 做静态检查，MySQL 实机用 `SHOW CREATE TABLE` 核对。
10. **排序规则**：DDL 未写 `COLLATE` 时 MySQL 8.0 默认 `utf8mb4_0900_ai_ci`；`db/schema.sql` 已显式写 `utf8mb4_unicode_ci`，实机已核对四张表。

### 15.1 Known limitations（明确不做本章修复）

1. **`query_faq` 字面 LIKE 漏召回**：`邮费` 查不到 `运费` 是预期行为，不是 bug。验收 3 只要求助手如实说明未查到并建议转人工；不得通过补种子数据“邮费”或让模型改写关键词来掩盖。下一步用同义词/向量检索解决。
2. **工具结果持久化截断**：超过 `TOOL_RESULT_MAX_CHARS = 4000` 的结果只存前 4000 字符；当前轮模型收到完整结果，后续从历史重建时看到截断版本。这是有意的存储保护。
3. **只做单轮工具调用**：模型返回多个工具调用时只执行第一个，其余丢弃并记录日志；不做 Agent Loop。

## 16. 交付物

- 代码：模型、仓储、工具、编排、路由、种子脚本、前端。
- 设计文档：本文件。
- 开发留痕：`dev-notes/ch02.md`（每阶段追记）。
- 测试：`uv run pytest` 结果；真实模型工具选择评估结果。
- 演示命令：README 中的 `docker compose` / `init_db` / `uvicorn` / `curl` 命令。
- 验收记录：三条浏览器验收问题与预期漏召回的实测结果。

## 附录 A：权威 DDL

以下内容与 `db/schema.sql` 一致；以文件为准。相比用户原始 DDL，仅补了每张表的 `COLLATE=utf8mb4_unicode_ci`，避免 MySQL 8.0 按字符集默认值落成 `utf8mb4_0900_ai_ci`。

```sql
-- =============================================================
-- ch02 · Function Calling 工具链 · 建表 DDL
-- 本章新建:faq / conversations / messages / tickets 四张表
-- 商品、订单、物流走工具内 mock,不建表
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4、COLLATE=utf8mb4_unicode_ci
-- 建表顺序:先 conversations,再依赖它的 messages / tickets
-- =============================================================

-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='人工工单';
```
