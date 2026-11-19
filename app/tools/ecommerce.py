"""mock 电商工具：不访问数据库，按 id 生成确定性数据。"""

import json
import random
from datetime import date, timedelta
from typing import Annotated, Any

from langchain_core.tools import tool
from pydantic import Field

_ORDER_STATUSES = ("待付款", "待发货", "已发货", "已签收", "已完成")
_SHIPPING_STATUSES = ("运输中", "派送中", "已签收")
_CARRIERS = ("顺丰速运", "中通快递", "圆通速递", "京东物流")
_PRODUCT_NAMES = (
    "无线蓝牙耳机",
    "便携充电宝",
    "智能手环",
    "桌面加湿器",
    "保温杯",
    "机械键盘",
)
_PRODUCT_CATEGORIES = ("数码配件", "生活电器", "家居日用", "运动户外")
_BASE_DATE = date(2026, 1, 1)


def _rng(prefix: str, identifier: str) -> random.Random:
    return random.Random(f"{prefix}:{identifier}")


def _ok(data: dict[str, Any]) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


@tool
def query_order(order_id: Annotated[str, Field(min_length=1)]) -> str:
    """根据订单号查询订单状态、商品明细和金额。"""
    rng = _rng("order", order_id)
    created_at = _BASE_DATE + timedelta(days=rng.randint(0, 180))
    quantity = rng.randint(1, 3)
    unit_price = round(rng.uniform(19, 499), 2)
    data = {
        "order_id": order_id,
        "status": rng.choice(_ORDER_STATUSES),
        "items": [
            {
                "product_id": f"P{rng.randint(100, 999)}",
                "name": rng.choice(_PRODUCT_NAMES),
                "quantity": quantity,
                "unit_price": unit_price,
            }
        ],
        "total_amount": round(unit_price * quantity, 2),
        "created_at": created_at.isoformat(),
    }
    return _ok(data)


@tool
def query_product(product_id: Annotated[str, Field(min_length=1)]) -> str:
    """根据商品编号查询商品名称、价格和库存。"""
    rng = _rng("product", product_id)
    data = {
        "product_id": product_id,
        "name": rng.choice(_PRODUCT_NAMES),
        "category": rng.choice(_PRODUCT_CATEGORIES),
        "price": round(rng.uniform(19, 599), 2),
        "stock": rng.randint(0, 500),
        "description": "商品详情以页面展示为准，支持 7 天无理由退换。",
    }
    return _ok(data)


@tool
def query_logistics(order_id: Annotated[str, Field(min_length=1)]) -> str:
    """根据订单号查询物流承运商、运单号和最新进度。"""
    rng = _rng("logistics", order_id)
    carrier = rng.choice(_CARRIERS)
    tracking_no = f"{rng.choice(('SF', 'ZT', 'YT', 'JD'))}{rng.randint(10**11, 10**12 - 1)}"
    shipped_at = _BASE_DATE + timedelta(days=rng.randint(0, 180))
    data = {
        "order_id": order_id,
        "carrier": carrier,
        "tracking_no": tracking_no,
        "status": rng.choice(_SHIPPING_STATUSES),
        "latest_event": rng.choice(
            (
                "快件已到达【杭州转运中心】",
                "快件正在派送途中，请保持电话畅通",
                "快件已签收，感谢使用",
            )
        ),
        "estimated_delivery": (shipped_at + timedelta(days=3)).isoformat(),
    }
    return _ok(data)
