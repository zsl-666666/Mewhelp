# 电商智能客服（Chapter 01：纯对话）

基于 Python 3.13 + FastAPI + LangChain 的电商客服后端。本章目标是跑通“纯对话”闭环：多轮对话、SSE 流式输出、Prompt 模板化、售后信息结构化提取，并附带一个最小聊天页用于浏览器验收。

本章不做工具调用、Agent 循环、会话持久化、鉴权与限流。

## 目录结构

```text
app/
  main.py              FastAPI 应用与路由（/、/health、/api/chat、/api/after-sales/extract）
  config.py            pydantic-settings 配置（读取 .env）
  llm.py               ChatOpenAI 构造，统一 OpenAI 兼容协议
  schemas.py           请求/响应模型与校验
  sse.py               SSE 帧格式化
  services/
    chat.py            流式对话与 Prompt 组装
    context.py         按 token 预算裁剪历史
    extract.py         售后信息结构化提取
  prompts/             客服 System Prompt 与售后抽取 Prompt
  static/index.html    最小聊天页
tests/                 离线单元测试（不访问网络）
evals/                 售后抽取标注样例与评估脚本
docs/                  Superpowers 设计与计划留痕
dev-notes/             开发过程记录
```

## 快速开始

前置要求：Python 3.13、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync
cp .env.example .env
# Windows cmd 用户请改用 copy .env.example .env（或直接使用 Git Bash）
# 编辑 .env，至少填写 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL
uv run uvicorn app.main:app --reload --port 8000
```

服务启动后：

- 聊天页：<http://127.0.0.1:8000/>
- 健康检查：`curl http://127.0.0.1:8000/health`，返回 `{"status":"ok","model":"..."}`

`.env` 主要配置项：

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `LLM_BASE_URL` | 是 | OpenAI 兼容上游地址 |
| `LLM_API_KEY` | 是 | 上游密钥；Ollama 可填任意非空值 |
| `LLM_MODEL` | 是 | 模型名 |
| `LLM_TEMPERATURE` | 否 | 默认 `0.3` |
| `LLM_MAX_TOKENS` | 否 | 默认 `1024` |
| `LLM_TIMEOUT_SECONDS` | 否 | 默认 `60` |
| `CONTEXT_TOKEN_BUDGET` | 否 | 历史裁剪预算，默认 `3000`（近似值） |
| `STRUCTURED_OUTPUT_METHOD` | 否 | `function_calling`（默认）/ `json_schema` |
| `COMPANY_NAME` / `SERVICE_HOURS` / `RETURN_WINDOW_DAYS` | 否 | 客服 Prompt 中的品牌与服务政策信息 |

## Provider 切换

应用侧统一使用 OpenAI 兼容协议，只改 `.env` 中的地址、密钥和模型名，不改代码。

| Provider | `LLM_BASE_URL` 示例 | `LLM_MODEL` 示例 | 说明 |
| --- | --- | --- | --- |
| GPT | `https://api.openai.com/v1` | `gpt-4o-mini` | 官方兼容端点 |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | 官方兼容端点 |
| Ollama | `http://127.0.0.1:11434/v1` | `qwen2.5:7b` | 本地服务；需支持工具调用才能使用默认结构化输出 |
| Claude（经网关） | `https://openrouter.ai/api/v1`（或 LiteLLM / one-api 网关地址） | `anthropic/claude-sonnet-4`（以网关为准） | 应用不直连 Anthropic，走 OpenAI 兼容网关 |

> 模型名以供应商当前文档为准；表中仅为示例，供应商可能调整或下线模型。

## API 与验收命令

接口是无状态的：客户端每轮发送完整 `messages`，最后一条必须是 `user`。

以下命令在 Git Bash / macOS / Linux 中执行；`-N` 用于关闭 curl 缓冲，便于观察流式输出。

### 1. 流式对话

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"我买的蓝牙耳机想退货，订单号 20261116001。"}]}'
```

预期看到多条增量帧，最后以结束帧收尾：

```text
data: {"delta": "好的"}
data: {"delta": "，已记录您的退货意向。"}
data: {"done": true, "trimmed": false}
```

`trimmed` 为 `true` 表示历史因 token 预算被裁剪。

流式过程中上游异常时，HTTP 状态仍为 200，但会发送 `event: error` 和 `data: {"code":"upstream_error","message":"..."}` 后关闭；下一页结构化提取接口的失败才是 502。

### 2. 第二轮对话（验证上下文）

先执行上一条命令，再把返回的 assistant 文本替换到下面的 `assistant` 字段，然后发送：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[
    {"role":"user","content":"我买的蓝牙耳机想退货，订单号 20261116001。"},
    {"role":"assistant","content":"（替换为第一轮的真实回复）"},
    {"role":"user","content":"那我应该怎么寄回？"}
  ]}'
```

预期第二轮回复能接住第一轮的退货意图与订单信息。

### 3. 结构化提取

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
- 回复通过 SSE 增量渲染，逐段出现在客服气泡中；
- 每次发送会带上当前会话的完整历史，可连续多轮对话；
- 流式期间禁用发送与清空按钮，出错或历史被裁剪时在状态栏提示。

该页面是最小演示，不保存会话、不渲染 Markdown、刷新后历史清空。

## 测试与评估

```bash
# 全量离线单元测试（不访问网络，不需要 .env）
uv run pytest

# 售后结构化提取评估（需要真实 .env 与可用上游）
uv run python -m evals.eval_structured
```

评估集位于 `evals/after_sales_cases.jsonl`，共 15 条标注样例。脚本逐条打印预期与实际，并按字段汇总：

- `order_id`：通过率需 ≥ 1.0
- `request_type`：通过率需 ≥ 0.8
- `expected_solution`：关键词全部命中率需 ≥ 0.8

未配置 `.env` 时脚本会提示缺少 `LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` 并以退出码 2 结束，不会伪造结果。

## 验收状态

截至本章交付，仓库内没有 `.env`，未连接真实模型，因此以下实机验收项仍为**待验**，不能视为已通过：

- 真实上游的流式增量输出；
- 第二轮回复对第一轮内容的引用；
- 浏览器聊天页的逐字渲染与多轮对话；
- 结构化提取的真实模型返回；
- `evals.eval_structured` 的通过率。

离线单元测试已通过（45 项）。配置 `.env` 后可按上文命令完成实机验收。

## 已知限制

- token 估算使用 LangChain 的近似计数，不是上游 tokenizer 的精确值；历史裁剪可能偏保守或偏宽松。
- 流式粒度由上游决定，可能一次返回多个 token，不保证严格逐字。
- 默认结构化输出方式为 `function_calling`；Ollama 模型需支持工具调用。切换 `json_schema` 时，需确认上游支持对应能力。
- 聊天页为最小演示，无登录、会话持久化、Markdown 渲染和移动端深度适配。
- 接口无鉴权与限流，仅适合本地开发与课程演示，不可直接暴露到公网。

## 安全提示

- `.env` 已被 `.gitignore` 忽略，**不要提交密钥**，也不要把密钥写进代码、测试、日志或截图。
- 上游服务会收到用户对话与售后文本，接入前应确认数据合规要求。
- 服务端错误会记录异常堆栈，但返回给客户端的错误文案已脱敏；生产环境仍需配置日志脱敏与访问控制。
