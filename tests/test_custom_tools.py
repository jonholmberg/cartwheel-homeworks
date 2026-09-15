"""The two custom HW1 tools, tested through their logic functions.

Everything runs offline: no LLM, no API keys. Expectations are derived from
the seeded world rather than hard-coded, so they follow the fixture data.
"""

from __future__ import annotations

from agent import db, tools
from agent.auth import AuthContext

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
SHOPPER_2 = AuthContext(user_id=2, role="shopper")
MERCHANT_STORE_1 = AuthContext(user_id=9001, role="merchant", store_id=1)
SUPPORT = AuthContext(user_id=9501, role="support")


# ---------------------------------------------------------------------------
# recommend_products: scope is baked in (no id argument), shopper-only.
# ---------------------------------------------------------------------------


def test_recommend_products_matches_bought_categories(world: dict) -> None:
    conn = db.connect()
    try:
        orders = db.list_orders_for_user(conn, 1)
        products = db.list_products(conn)
    finally:
        conn.close()

    by_id = {p.id: p for p in products}
    owned_ids = {o.product_id for o in orders}
    owned_categories = {
        by_id[o.product_id].category for o in orders if o.product_id in by_id
    }
    expected = [
        p for p in products if p.category in owned_categories and p.id not in owned_ids
    ]
    expected.sort(key=lambda p: (p.price_usd, p.id))
    expected_ids = [p.id for p in expected[:5]]

    result = tools.recommend_products(SHOPPER_1, limit=5)
    assert result["ok"] is True
    assert result["count"] == len(result["products"])
    assert [p["product_id"] for p in result["products"]] == expected_ids
    # Never recommend something already owned, and stay within bought categories.
    assert all(p["product_id"] not in owned_ids for p in result["products"])
    assert all(
        by_id[p["product_id"]].category in owned_categories
        for p in result["products"]
    )
    # Deterministic ordering: price ascending, then product_id ascending.
    keys = [(p["price_usd"], p["product_id"]) for p in result["products"]]
    assert keys == sorted(keys)


def test_recommend_products_clamps_limit(world: dict) -> None:
    unclamped = tools.recommend_products(SHOPPER_1, limit=999)
    assert unclamped["ok"] is True
    assert unclamped["count"] <= tools.MAX_SEARCH_LIMIT

    at_least_one = tools.recommend_products(SHOPPER_1, limit=0)
    assert at_least_one["ok"] is True
    assert len(at_least_one["products"]) <= 1


def test_recommend_products_is_shopper_only(world: dict) -> None:
    for ctx in (MERCHANT_STORE_1, SUPPORT):
        result = tools.recommend_products(ctx)
        assert result["ok"] is False
        assert result["error"] == "invalid_argument"


# ---------------------------------------------------------------------------
# order_summary: takes an order_id, so it must enforce the access matrix.
# ---------------------------------------------------------------------------


def test_order_summary_happy_path(world: dict) -> None:
    conn = db.connect()
    try:
        title = conn.execute(
            "SELECT p.title FROM orders o JOIN products p ON o.product_id = p.id "
            "WHERE o.id = 4127"
        ).fetchone()[0]
    finally:
        conn.close()

    result = tools.order_summary(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["order_id"] == 4127
    assert title in result["summary"]
    assert result["status"] in result["summary"]


def test_order_summary_denies_other_shopper_scope_first(world: dict) -> None:
    # An existing order the caller does not own returns permission_denied,
    # never not_found, so a stranger cannot tell the order exists.
    result = tools.order_summary(SHOPPER_2, 4127)
    assert result["ok"] is False
    assert result["error"] == "permission_denied"


def test_order_summary_allows_support(world: dict) -> None:
    result = tools.order_summary(SUPPORT, 4127)
    assert result["ok"] is True
    assert result["order_id"] == 4127


def test_order_summary_missing_order_is_not_found(world: dict) -> None:
    result = tools.order_summary(SHOPPER_1, 999_999)
    assert result["ok"] is False
    assert result["error"] == "not_found"
