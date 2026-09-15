"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from typing import Any

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {
        "ok": False,
        "error": "not_found",
        "reason": f"no policy doc has id {policy_id!r}",
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    tokens = query.split()
    if not tokens:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "query must be non-empty",
        }
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "max_price_usd must be strictly positive",
        }

    lowered = [t.lower() for t in tokens]
    with db.connection() as conn:
        store_id = None
        if store is not None:
            matched = db.get_store_by_name(conn, store)
            if matched is None:
                return {
                    "ok": False,
                    "error": "not_found",
                    "reason": f"no store matches {store!r}",
                }
            store_id = matched.id
        candidates = db.list_products(conn, store_id)

    matches = []
    for product in candidates:
        haystack = f"{product.title}\n{product.description}".lower()
        if not all(token in haystack for token in lowered):
            continue
        if max_price_usd is not None and product.price_usd > max_price_usd:
            continue
        matches.append(product)

    matches.sort(key=lambda p: (p.price_usd, p.id))
    clamped = max(1, min(limit, MAX_SEARCH_LIMIT))
    products = [
        {
            "product_id": p.id,
            "store_id": p.store_id,
            "title": p.title,
            "price_usd": p.price_usd,
        }
        for p in matches[:clamped]
    ]
    return {"ok": True, "products": products, "count": len(products)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "support staff have no orders of their own; use get_order",
        }
    with db.connection() as conn:
        if ctx.role == "merchant":
            orders = db.list_orders_for_store(conn, ctx.store_id)
        else:
            orders = db.list_orders_for_user(conn, ctx.user_id)
    public = [o.to_public_dict() for o in orders[:DEFAULT_ORDER_LIMIT]]
    return {"ok": True, "orders": public, "count": len(public)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {
                "ok": False,
                "error": "not_found",
                "reason": f"no order has id {order_id}",
            }
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"{ctx.role} may not cancel order {order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": (
                    f"order {order_id} is {order.status!r}; orders can be "
                    "cancelled only before shipment"
                ),
            }
        db.set_order_status(conn, order_id, "cancelled")
    return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    import difflib

    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id)
        elif ctx.role == "merchant":
            orders = db.list_orders_for_store(conn, ctx.store_id)
        else:  # support: search across all orders
            ids = [row[0] for row in conn.execute("SELECT id FROM orders").fetchall()]
            orders = [db.get_order(conn, oid) for oid in ids]
        titles = {p.id: p.title for p in db.list_products(conn)}

    q = query.lower().strip()
    q_tokens = {t for t in q.split() if len(t) >= 3}
    scored: list[tuple[float, Any]] = []
    for order in orders:
        title = (titles.get(order.product_id) or "").lower()
        if not title:
            continue
        title_tokens = {t for t in title.split() if len(t) >= 3}
        ratio = difflib.SequenceMatcher(None, q, title).ratio()
        if q in title or title in q:
            score = 1.0
        elif q_tokens & title_tokens:
            score = max(ratio, 0.6)
        elif ratio >= 0.6:
            score = ratio
        else:
            continue
        scored.append((score, order))

    scored.sort(key=lambda pair: (-pair[0], -pair[1].id))
    orders_out = [order.to_public_dict() for _, order in scored[:5]]
    return {"ok": True, "orders": orders_out}


def recommend_products(ctx: AuthContext, limit: int = 5) -> dict[str, Any]:
    """Recommend catalog products from the caller's own purchase history.

    Custom tool (HW1 Part A). Risk tier: read.

    Scope is baked in: this tool never takes an order or user id, so it can
    only ever read the authenticated caller's own orders via
    agent.db.list_orders_for_user(conn, ctx.user_id). There is no id a caller
    could supply to reach another person's history, so no permission check is
    needed. Only shoppers have a personal purchase history, so merchants and
    support get an invalid_argument error.

    Recommendation rule (deterministic, explainable): take the categories of
    the products the caller has already bought, then suggest other products in
    those same categories that the caller has not already ordered.

    Args:
        ctx: The caller's auth context.
        limit: Maximum products to return. Clamped to [1, MAX_SEARCH_LIMIT].

    Returns:
        {"ok": True, "products": [{product_id, store_id, title, price_usd}],
        "count": N} sorted by price_usd ascending then product_id ascending.
        A caller with no purchase history gets an empty list, still ok.
        Non-shoppers get {"ok": False, "error": "invalid_argument", ...}.
    """
    if ctx.role != "shopper":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": (
                "recommendations are based on your own purchase history; "
                "only shoppers have one"
            ),
        }
    with db.connection() as conn:
        orders = db.list_orders_for_user(conn, ctx.user_id)
        products = db.list_products(conn)

    by_id = {p.id: p for p in products}
    owned_ids = {o.product_id for o in orders}
    owned_categories = {
        by_id[o.product_id].category for o in orders if o.product_id in by_id
    }
    recommended = [
        p
        for p in products
        if p.category in owned_categories and p.id not in owned_ids
    ]
    recommended.sort(key=lambda p: (p.price_usd, p.id))
    clamped = max(1, min(limit, MAX_SEARCH_LIMIT))
    out = [
        {
            "product_id": p.id,
            "store_id": p.store_id,
            "title": p.title,
            "price_usd": p.price_usd,
        }
        for p in recommended[:clamped]
    ]
    return {"ok": True, "products": out, "count": len(out)}


def order_summary(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Return a one-line human summary of one order. Risk tier: read.

    Custom tool (HW1 Part A). Unlike recommend_products, this tool takes an
    order_id the caller supplies, so it must enforce the access matrix. The
    scope check runs before anything is revealed, so an out-of-scope caller
    cannot distinguish "order exists but is not yours" from "no such order"
    (the IDOR / enumeration protection that cancel_order also relies on).

    Args:
        ctx: The caller's auth context.
        order_id: The order to summarize.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found", ...}.
        If the order is out of the caller's scope:
        agent.auth.permission_denied(...).
        On success: {"ok": True, "order_id": int, "status": str,
        "summary": str} where summary joins quantity, product title, total,
        status, and order date into one sentence.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {
                "ok": False,
                "error": "not_found",
                "reason": f"no order has id {order_id}",
            }
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"{ctx.role} may not view order {order_id}"
            )
        product = next(
            (p for p in db.list_products(conn) if p.id == order.product_id), None
        )

    title = product.title if product is not None else "unknown product"
    summary = (
        f"Order {order.id}: {order.quantity} x {title} "
        f"for ${order.total_usd:.2f}, status {order.status}, "
        f"ordered {order.ordered_at.isoformat()}"
    )
    return {
        "ok": True,
        "order_id": order.id,
        "status": order.status,
        "summary": summary,
    }
