from typing import Literal

from pydantic import BaseModel, Field, field_validator

RequestType = Literal["退款", "退货", "换货", "维修", "物流", "咨询", "其他"]


class ChatRequest(BaseModel):
    user_id: str = Field(default="demo-user", min_length=1, max_length=64)
    conversation_id: int | None = Field(default=None, gt=0)
    message: str = Field(min_length=1)

    @field_validator("user_id", "message")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class AfterSalesRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class AfterSalesExtraction(BaseModel):
    order_id: str | None = Field(
        default=None,
        description="订单号；文本未明确出现时为 null，禁止猜测",
    )
    request_type: RequestType = Field(description="诉求类型")
    expected_solution: str | None = Field(
        default=None,
        description="用户明确表达的期望方案；未表达时为 null",
    )

    @field_validator("order_id", "expected_solution", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value
