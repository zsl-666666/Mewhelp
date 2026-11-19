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
