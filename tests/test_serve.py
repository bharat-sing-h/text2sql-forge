from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

pytest.importorskip("fastapi")

from t2sforge.serve.app import Settings, create_app
from t2sforge.serve.backend import BackendError, ChatBackend
from t2sforge.serve.pipeline import DatabaseRegistry, UnknownDatabaseError

BACKEND = "http://model.test/v1"


def completion(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 12},
        },
    )


@pytest.fixture
def client(shop_db: Path) -> httpx.AsyncClient:
    settings = Settings(
        db_root=str(shop_db.parent.parent),
        backend_url=BACKEND,
        model="sql",
        max_repairs=1,
        max_rows=3,
    )
    app = create_app(settings, ChatBackend(BACKEND, "sql", max_retries=1))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")


@respx.mock
async def test_query_happy_path(client: httpx.AsyncClient) -> None:
    route = respx.post(f"{BACKEND}/chat/completions").mock(
        return_value=completion("```sql\nSELECT name FROM customers WHERE city = 'Delhi';\n```")
    )
    resp = await client.post("/v1/query", json={"db_id": "shop", "question": "Who is in Delhi?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sql"] == "SELECT name FROM customers WHERE city = 'Delhi'"
    assert sorted(r[0] for r in body["rows"]) == ["Asha", "Chen"]
    assert body["error"] is None
    assert not body["repaired"]
    sent = json.loads(route.calls.last.request.content)
    assert sent["temperature"] == 0.0
    assert "CREATE TABLE customers" in sent["messages"][1]["content"]  # schema prompt
    metrics = (await client.get("/metrics")).text
    assert 't2s_requests_total{outcome="ok"} 1.0' in metrics


@respx.mock
async def test_self_correction_after_execution_error(client: httpx.AsyncClient) -> None:
    route = respx.post(f"{BACKEND}/chat/completions").mock(
        side_effect=[
            completion("SELECT nme FROM customers"),
            completion("SELECT name FROM customers"),
        ]
    )
    body = (await client.post("/v1/query", json={"db_id": "shop", "question": "All names?"})).json()
    assert body["repaired"] is True
    assert [a["stage"] for a in body["attempts"]] == ["execute", "ok"]
    assert body["truncated"] is True
    assert len(body["rows"]) == 3
    repair_prompt = json.loads(route.calls.last.request.content)["messages"]
    assert "no such column: nme" in repair_prompt[-1]["content"]
    assert repair_prompt[-2] == {"role": "assistant", "content": "SELECT nme FROM customers"}
    assert 't2s_repairs_total{trigger="execute"} 1.0' in (await client.get("/metrics")).text


@respx.mock
async def test_guard_blocks_destructive_sql(client: httpx.AsyncClient, shop_db: Path) -> None:
    respx.post(f"{BACKEND}/chat/completions").mock(return_value=completion("DELETE FROM orders"))
    body = (await client.post("/v1/query", json={"db_id": "shop", "question": "Clean up"})).json()
    assert body["error"].startswith("rejected by SQL guard")
    assert [a["stage"] for a in body["attempts"]] == ["guard", "guard"]
    from t2sforge.sql import run_query

    assert run_query(shop_db, "SELECT COUNT(*) FROM orders").rows == [(5,)]
    assert 't2s_requests_total{outcome="failed_guard"} 1.0' in (await client.get("/metrics")).text


@respx.mock
async def test_generate_only_and_validation(client: httpx.AsyncClient) -> None:
    respx.post(f"{BACKEND}/chat/completions").mock(return_value=completion("SELECT 1"))
    body = (
        await client.post("/v1/query", json={"db_id": "shop", "question": "One?", "execute": False})
    ).json()
    assert body["sql"] == "SELECT 1"
    assert body["rows"] == []
    assert (
        await client.post("/v1/query", json={"db_id": "shop", "question": "x"})
    ).status_code == 422
    assert (
        await client.post("/v1/query", json={"db_id": "../etc", "question": "hello"})
    ).status_code == 404
    assert (
        await client.post("/v1/query", json={"db_id": "nope", "question": "hello"})
    ).status_code == 404


@respx.mock
async def test_backend_failure_maps_to_502(client: httpx.AsyncClient) -> None:
    respx.post(f"{BACKEND}/chat/completions").mock(return_value=httpx.Response(503))
    resp = await client.post("/v1/query", json={"db_id": "shop", "question": "Anything?"})
    assert resp.status_code == 502
    respx.post(f"{BACKEND}/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    assert (
        await client.post("/v1/query", json={"db_id": "shop", "question": "Anything?"})
    ).status_code == 502


@respx.mock
async def test_health_ready_and_schema(client: httpx.AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    respx.get(f"{BACKEND}/models").mock(return_value=httpx.Response(200, json={"data": []}))
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["databases"] == 1
    respx.get(f"{BACKEND}/models").mock(side_effect=httpx.ConnectError("down"))
    assert (await client.get("/readyz")).status_code == 503
    assert (await client.get("/v1/databases")).json() == {"databases": ["shop"]}
    schema = (await client.get("/v1/databases/shop/schema")).json()["schema"]
    assert "CREATE TABLE orders" in schema
    assert (await client.get("/v1/databases/missing/schema")).status_code == 404


def test_registry_rejects_traversal(shop_db: Path, tmp_path: Path) -> None:
    reg = DatabaseRegistry(shop_db.parent.parent)
    assert reg.path("shop") == shop_db.resolve()
    for bad in ["../shop", "shop/../shop", "", "a" * 200]:
        with pytest.raises(UnknownDatabaseError):
            reg.path(bad)
    assert DatabaseRegistry(tmp_path / "missing").list() == []


@respx.mock
async def test_backend_retries_then_succeeds() -> None:
    respx.post(f"{BACKEND}/chat/completions").mock(
        side_effect=[httpx.Response(429), completion("SELECT 2")]
    )
    backend = ChatBackend(BACKEND, "sql", max_retries=2)
    assert (await backend.complete([{"role": "user", "content": "q"}])).text == "SELECT 2"
    respx.post(f"{BACKEND}/chat/completions").mock(return_value=httpx.Response(400))
    with pytest.raises(BackendError, match="400"):
        await backend.complete([{"role": "user", "content": "q"}])
    await backend.aclose()
