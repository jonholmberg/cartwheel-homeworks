"""Authentication tests for the Homework 2 endpoint.

These run fully offline: they exercise session creation and token
authorization directly, without Langfuse, Docker, or a model provider key.
Tracing itself is checked by inspecting recorded spans in Part E.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


def test_create_session_rejects_role_mismatch(world: dict) -> None:
    """A claimed role that differs from the stored role is refused (403).

    User 1 is a shopper in the seeded world. Claiming the merchant role must
    not bind a merchant context; the server trusts the database, not the
    request.
    """
    server_app._SESSIONS.clear()
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(
            server_app.SessionCreate(user_id=1, role="merchant")
        )
    assert exc.value.status_code == 403
    assert server_app._SESSIONS == {}


def test_token_cannot_authorize_a_different_session(world: dict) -> None:
    """A token minted for one session must not authorize another (403).

    The token carries its own session_id; `_authorize` rejects it when the
    path names a different session, even though the signature is valid.
    """
    server_app._SESSIONS.clear()
    first = server_app.create_session(
        server_app.SessionCreate(user_id=1, role="shopper")
    )
    second = server_app.create_session(
        server_app.SessionCreate(user_id=9002, role="merchant")
    )

    # The first token is valid for its own session.
    ctx = server_app._authorize(
        first["session_id"], f"Bearer {first['token']}"
    )
    assert ctx.user_id == 1

    # But it must not authorize the second session.
    with pytest.raises(HTTPException) as exc:
        server_app._authorize(
            second["session_id"], f"Bearer {first['token']}"
        )
    assert exc.value.status_code == 403
