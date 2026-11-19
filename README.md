# 电商智能客服（Chapter 02：Function Calling）

基于 Python 3.13 + FastAPI + LangChain 的电商客服后端。本章在纯对话闭环之上接入 **单轮 Function Calling**：模型自己决定调用订单、商品、物流、FAQ 或工单工具，工具结果回灌模型后继续流式生成最终回答；会话与消息由服务端持久化到 MySQL。

本章只做单轮工具调用（每次用户消息最多执行一个工具），不做多轮 Agent Loop，也不做 RAG。

## 目录结构

```text
app/
  main.py                  FastAPI 应用与路由（/、/health、/api/chat、/api/after-sales/extract）
  config.py                pydantic-settings 配置（读取 .env）
  db.py                    SQLAlchemy Engine / Session 基础设施
  llm.py                   ChatOpenAI 构造，统一 OpenAI 兼容协议
  models.py                ORM：conversations / messages / faq / tickets
  schemas.py               请求/响应模型与校验
  sse.py                   SSE 帧格式化
  repositories/            会话、FAQ、工单仓储
  tools/
    registry.py            工具注册、参数校验、超时重试与执行事件流
    ecommerce.py           query_order / query_product / query_logistics（确定性 mock）
    knowledge.py           query_faq（SQL LIKE）
    tickets.py             create_ticket（写 tickets 并转人工）
  services/
    chat.py                单轮工具编排、SSE 输出、消息落库
    context.py             历史裁剪与工具对修复
    extract.py             售后信息结构化提取
  prompts/                 客服 System Prompt、工具使用规则、售后抽取 Prompt
  seed_data.py             FAQ 种子（10 条）
  static/index.html        聊天页（服务端管历史 + 工具徽章）
db/schema.sql              四张表的 MySQL 权威 DDL
scripts/init_db.py         幂等建表 + FAQ 种子脚本
evals/
  tool_calling_cases.jsonl 工具选择标注样例（10 条）
  eval_tool_calling.py     工具选择评估；只评分，不执行工具
tests/                     离线单元测试（不访问网络）
docs/                      Superpowers 设计与计划留痕
dev-notes/                 开发过程记录
```

## 快速开始

前置要求：Python 3.13、[uv](https://docs.astral.sh/uv/)、Docker Desktop（或兼容的 Docker + Compose）。

```bash
uv sync
cp .env.example .env
# Windows cmd 用户请改用 copy .env.example .env（或直接使用 Git Bash）
# 编辑 .env：至少填写 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
#         以及 MYSQL_ROOT_PASSWORD、MYSQL_PASSWORD（不要提交 .env）

docker compose up -d
uv run python -m scripts.init_db
uv run uvicorn app.main:app --reload --port 8000
```

`init_db` 是幂等的：

- 四张表都不存在：执行 `db/schema.sql` 建表；
- 四张表齐全：跳过建表；
- 结构不完整：直接报错，不自动补建、不删表；
- `faq` 为空时灌入 10 条种子，已有数据时不重复写入。

服务启动后：

- 聊天页：<http://127.0.0.1:8000/>
- 健康检查：`curl http://127.0.0.1:8000/health`，返回 `{"status":"ok","model":"..."}`

### `.env` 主要配置项

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | 是 | OpenAI 兼容上游地址 |
| `LLM_API_KEY` | 是 | 上游密钥；Ollama 可填任意非空值 |
| `LLM_MODEL` | 是 | 模型名；必须支持工具调用 |
| `LLM_TEMPERATURE` | 否 | 默认 `0.3` |
| `LLM_MAX_TOKENS` | 否 | 默认 `1024` |
| `LLM_TIMEOUT_SECONDS` | 否 | 默认 `60` |
| `CONTEXT_TOKEN_BUDGET` | 否 | 历史裁剪预算，默认 `3000`（近似值） |
| `TOOL_TIMEOUT_SECONDS` | 否 | 单次工具超时，默认 `5` |
| `TOOL_MAX_RETRIES` | 否 | 可重试工具的最大重试次数，默认 `2` |
| `TOOL_RETRY_DELAY_SECONDS` | 否 | 重试间隔，默认 `0.2` |
| `TOOL_RESULT_MAX_CHARS` | 否 | 工具结果持久化截断长度，默认 `4000` |
| `MYSQL_HOST` / `MYSQL_PORT` / `MYSQL_DATABASE` | 是 | 本地 MySQL 连接信息 |
| `MYSQL_USER` / `MYSQL_PASSWORD` | 是 | 应用数据库账号；密码只放在本机 `.env` |
| `MYSQL_ROOT_PASSWORD` | 是 | 仅供容器初始化使用 |
| `COMPANY_NAME` / `SERVICE_HOURS` / `RETURN_WINDOW_DAYS` | 否 | 客服 Prompt 中的品牌与服务政策信息 |

## API

### `POST /api/chat`

请求体（服务端管历史，破坏性变更）：

```json
{
  "user_id": "demo-user",
  "conversation_id": 12,
  "message": "订单 1001 的物流到哪了"
}
```

- `user_id`：可选，默认 `"demo-user"`，1–64 字符。
- `conversation_id`：可选，正整数；不存在返回 404。
- `message`：必填，非空白；空白返回 422。
- 不传 `conversation_id` 时新建会话；新建的 id 通过首帧 `meta` 返回，后续请求带上它即可续聊。

正常 SSE 帧顺序：

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

工具重试/超时/失败帧：

```text
event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "retrying", "attempt": 2, "max_attempts": 3, "error_code": "timeout", "error": "工具执行超时"}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "timeout", "ok": false, "error_code": "timeout", "error": "工具执行超时", "retryable": true, "attempts": 3, "duration_ms": 5003}

event: tool
data: {"tool_call_id": "call_abc", "name": "query_logistics", "label": "查物流", "status": "failed", "ok": false, "error_code": "validation_error", "error": "参数校验失败", "retryable": false, "attempts": 0, "duration_ms": 0}
```

- 每个工具调用必须以 `completed` / `timeout` / `failed` 收尾，不会只停留在 `running`。
- 工具失败不会中断 SSE，也不走 `event: error`；错误结果会回灌模型，模型继续回答。
- 只有模型/上游失败才发 `event: error`，HTTP 状态仍为 200：

```text
event: error
data: {"code": "upstream_error", "message": "模型服务暂时不可用，请稍后重试"}
```

### 验收命令

以下命令在 Git Bash / macOS / Linux 中执行；`-N` 关闭 curl 缓冲。

> Windows Git Bash 下如果中文请求体被错误编码，先把 JSON 保存为 UTF-8 文件，再用 `--data-binary @文件名` 发送。

```bash
# 1. 物流查询：应出现 tool running / completed，回答基于 mock 物流数据
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"订单 1001 的物流到哪了"}'

# 2. FAQ 命中：应选中 query_faq，回答退货政策
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"退货政策是什么"}'

# 3. FAQ 关键词漏召回（known limitation）：选中 query_faq + keyword=邮费，但 LIKE 查不到
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user","message":"邮费是多少"}'
```

### `POST /api/after-sales/extract`

```bash
curl -X POST http://127.0.0.1:8000/api/after-sales/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"订单 20261116001 的耳机坏了，我想退货退款，希望明天上门取件。"}'
```

返回三个字段（示例，具体措辞由模型生成）：

```json
{
  "order_id": "20261116001",
  "request_type": "退货",
  "expected_solution": "希望明天上门取件"
}
```

`order_id` 未在文本中明确出现时为 `null`；`expected_solution` 未被用户明确表达时为 `null`；`request_type` 取值为 退款 / 退货 / 换货 / 维修 / 物流 / 咨询 / 其他。输入不合法返回 422，上游提取失败返回 502。

## 聊天页

启动服务后打开 <http://127.0.0.1:8000/>：

- 用户与客服消息以左右气泡展示；
- 回复通过 SSE 增量渲染；
- 服务端保存会话历史，页面只在本地保存 `conversation_id`，后续消息自动续聊；
- 工具执行时显示徽章：`正在查物流…` → `查物流`；重试显示 `重试中（2/3）…`，超时/失败显示红色错误样式；
- 流式期间禁用发送与清空按钮，出错或历史被裁剪时在状态栏提示；
- 点击“清空对话”会重置本地 `conversation_id`，下一轮创建新会话；刷新页面同样从新会话开始。

该页面是最小演示，不渲染 Markdown；刷新后不恢复历史。

## 测试与评估

```bash
# 全量离线单元测试（不访问网络，不需要 .env）
uv run pytest

# 工具选择评估（需要真实 .env 与支持工具调用的上游）
uv run python -m evals.eval_tool_calling

# 售后结构化提取评估（需要真实 .env 与可用上游）
uv run python -m evals.eval_structured
```

工具选择评估集位于 `evals/tool_calling_cases.jsonl`，共 10 条，覆盖订单、物流、商品、FAQ 与三类工单。评分只检查模型选择与模型可见参数，不执行工具、不建表、不写库；通过率阈值 `0.8`。未配置 `.env` 时脚本以退出码 2 结束，不伪造结果。

售后提取评估集位于 `evals/after_sales_cases.jsonl`，共 15 条标注样例，按字段汇总通过率。

### 当前验收状态（2026-11-19）

- 全量离线测试：**90 passed**。
- 工具选择评估：**10/10（100%）**。
- MySQL 建表与 FAQ 种子：已完成，`faq_count=10`，四张表齐全。
- 三条 curl 与浏览器徽章验收：**待重跑**。本机建表时三个中文 ENUM 被双重编码，导致 `POST /api/chat` 返回 `DataError (1265, "Data truncated for column 'status'")`；修复需要执行不可逆 `ALTER TABLE`，尚未获得授权，未做任何数据修改。详细诊断见 `dev-notes/ch02.md` 阶段 23。

## 数据库

四张表的权威 DDL 在 `db/schema.sql`：

- `conversations`：会话壳，`status` 为 进行中 / 已转人工 / 已结束；
- `messages`：消息流水，`role` 为 user / assistant / tool；`tool` 行的 `content` 是自描述 JSON envelope；
- `faq`：FAQ 问答对，`query_faq` 的数据源；
- `tickets`：人工工单，`ticket_type` 为 售后 / 投诉 / 咨询。

`messages` 的 tool 行落库约定：`tool_call_id` 写列，`tool_calls` 为 NULL，`content` 为

```json
{
  "tool": "query_logistics",
  "arguments": {"order_id": "1001"},
  "ok": true,
  "result": "{\"ok\":true,\"data\":{...}}",
  "truncated": false,
  "original_chars": 58,
  "attempts": 1,
  "duration_ms": 12
}
```

模型当前轮拿到完整工具结果；持久化时 `result` 最多保存 `TOOL_RESULT_MAX_CHARS` 字符，超出则 `truncated=true` 并记录 `original_chars`。

## 已知限制

- `query_faq` 使用字面 `LIKE`：用户问“邮费是多少”时模型按原词检索，但 FAQ 种子只有“运费怎么计算？”，因此会漏召回。这是 **known limitation**，不是 bug；下一章引入检索/向量方案解决。
- 工具结果持久化会截断到 4000 字符；后续从历史重建时只能看到截断版本。
- 只做单轮工具调用：模型一次最多选一个工具，拿到结果后直接收敛，不继续调用第二个工具。
- token 估算使用 LangChain 的近似计数，不是上游 tokenizer 的精确值。
- 流式粒度由上游决定，可能一次返回多个 token，不保证严格逐字。
- 默认结构化输出方式为 `function_calling`；Ollama 模型需支持工具调用。
- 聊天页为最小演示，无登录、无 Markdown 渲染、无移动端深度适配。
- 接口无鉴权与限流，仅适合本地开发与课程演示，不可直接暴露到公网。

## 安全提示

- `.env` 已被 `.gitignore` 忽略，**不要提交密钥**，也不要把密钥写进代码、测试、日志或截图。
- 上游服务会收到用户对话与售后文本，接入前应确认数据合规要求。
- 服务端错误会记录异常堆栈，但返回给客户端的错误文案已脱敏；生产环境仍需配置日志脱敏与访问控制。
