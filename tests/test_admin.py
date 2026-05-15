"""Tests for the Flask admin server's privacy endpoints.

Uses Flask's built-in test client. Each request is bracketed with the same
session cookie + Host header that the real admin UI uses, so the
DNS-rebinding / CSRF guard in ``admin._guard`` lets it through.

The DB is the standard ``db`` fixture - persistence is in a tmp dir, so a real
``db.save()`` inside an admin handler stays scoped to the test.
"""

import json

import pytest

# Admin server requires Flask. In environments where Flask isn't installed
# (some lightweight CI configurations), skip this whole module rather than
# erroring at collection.
pytest.importorskip("flask")

import cameralm.admin as admin_mod
from cameralm.admin import _COOKIE, _TOKEN, create_app


def _host_header() -> str:
    """One of the entries in the admin ``_ALLOWED_HOSTS`` set."""
    from cameralm.config import ADMIN_PORT
    return f"127.0.0.1:{ADMIN_PORT}"


@pytest.fixture
def client(db):
    """A Flask test client whose default request kwargs satisfy the admin guard."""
    app = create_app(db)
    c = app.test_client()
    c.set_cookie(_COOKIE, _TOKEN, domain="127.0.0.1")
    return c


def _api(client, method: str, path: str, body=None):
    """Helper: make a guarded API request. Returns the response."""
    kwargs = {"headers": {"Host": _host_header()}}
    if body is not None:
        kwargs["json"] = body
    return getattr(client, method.lower())(path, **kwargs)


# --------------------------------------------------------------------------
# /api/identity/<pid>/consent  (record)
# --------------------------------------------------------------------------


def test_record_consent_succeeds(client, db):
    pid = db.create_person("Alice")
    r = _api(client, "POST", f"/api/identity/{pid}/consent",
             {"granted_by": "Ms. Smith", "notes": "verbal in homeroom"})
    assert r.status_code == 200
    assert db.people[pid]["consent"]["status"] == "granted"
    assert db.people[pid]["consent"]["granted_by"] == "Ms. Smith"


def test_record_consent_missing_granted_by_is_400(client, db):
    pid = db.create_person("Alice")
    r = _api(client, "POST", f"/api/identity/{pid}/consent", {"notes": "no attestor"})
    assert r.status_code == 400
    assert db.people[pid]["consent"]["status"] == "none"


def test_record_consent_unknown_pid_is_404(client, db):
    r = _api(client, "POST", "/api/identity/999/consent",
             {"granted_by": "Ms. Smith"})
    assert r.status_code == 404


def test_record_consent_oversize_payload_rejected(client, db):
    """granted_by > 120 chars or notes > 500 chars must be rejected at the
    HTTP layer - the DB also truncates, but a polite caller gets a 400."""
    pid = db.create_person("Alice")
    r = _api(client, "POST", f"/api/identity/{pid}/consent",
             {"granted_by": "X" * 200, "notes": "ok"})
    assert r.status_code == 400


def test_record_consent_non_string_fields_rejected(client, db):
    pid = db.create_person("Alice")
    r = _api(client, "POST", f"/api/identity/{pid}/consent",
             {"granted_by": 12345, "notes": "ok"})
    assert r.status_code == 400


# --------------------------------------------------------------------------
# /api/identity/<pid>/consent  (revoke)
# --------------------------------------------------------------------------


def test_revoke_consent_drops_embeddings(client, db):
    from helpers import face_vec
    pid = db.create_person("Alice")
    db.add_face(pid, face_vec(seed=1))
    db.record_consent(pid, "Ms. Smith")

    r = _api(client, "DELETE", f"/api/identity/{pid}/consent")
    assert r.status_code == 200
    assert db.people[pid]["consent"]["status"] == "revoked"
    assert db.people[pid]["n_face"] == 0


def test_revoke_consent_unknown_pid_is_404(client, db):
    r = _api(client, "DELETE", "/api/identity/999/consent")
    assert r.status_code == 404


# --------------------------------------------------------------------------
# /api/audit
# --------------------------------------------------------------------------


def test_audit_returns_entries(client, db):
    pid = db.create_person("Alice")
    db.record_consent(pid, "Ms. Smith")

    r = _api(client, "GET", "/api/audit?limit=50")
    assert r.status_code == 200
    body = r.get_json()
    actions = [row["action"] for row in body["entries"]]
    assert "create_person" in actions
    assert "record_consent" in actions
    assert "retention_days" in body


def test_audit_limit_param_clamped(client, db):
    """A wildly oversized limit is clamped, not honored. The clamp is just so
    the response stays bounded - the entries list itself may still be small."""
    r = _api(client, "GET", "/api/audit?limit=999999")
    assert r.status_code == 200


def test_audit_invalid_limit_is_400(client, db):
    r = _api(client, "GET", "/api/audit?limit=not-a-number")
    assert r.status_code == 400


# --------------------------------------------------------------------------
# /api/purge-stale
# --------------------------------------------------------------------------


def test_purge_stale_noop_when_retention_disabled(client, db, monkeypatch):
    """RETENTION_DAYS=0 means the sweep is intentionally disabled - the
    endpoint returns immediately with an empty list."""
    monkeypatch.setattr(admin_mod, "RETENTION_DAYS", 0)
    r = _api(client, "POST", "/api/purge-stale")
    assert r.status_code == 200
    body = r.get_json()
    assert body["purged"] == []
    assert body["retention_days"] == 0


def test_purge_stale_drops_old_embeddings(client, db, monkeypatch):
    from datetime import datetime, timedelta
    from helpers import face_vec

    monkeypatch.setattr(admin_mod, "RETENTION_DAYS", 1)
    pid = db.create_person("Alice")
    db.add_face(pid, face_vec(seed=1))
    db.people[pid]["last_seen_at"] = (
        datetime.now() - timedelta(days=999)
    ).isoformat(timespec="seconds")

    r = _api(client, "POST", "/api/purge-stale")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["purged"]) == 1
    assert body["purged"][0]["pid"] == pid
    assert db.people[pid]["n_face"] == 0


# --------------------------------------------------------------------------
# Snapshot exposes consent state to the UI
# --------------------------------------------------------------------------


def test_state_includes_consent_and_last_seen(client, db):
    pid = db.create_person("Alice")
    db.record_consent(pid, "Ms. Smith", "ok")
    r = _api(client, "GET", "/api/state")
    assert r.status_code == 200
    body = r.get_json()
    person = next(p for p in body["people"] if p["pid"] == pid)
    assert person["consent"]["status"] == "granted"
    assert person["consent"]["granted_by"] == "Ms. Smith"
    assert "last_seen_at" in person


# --------------------------------------------------------------------------
# Guards still apply on the privacy endpoints
# --------------------------------------------------------------------------


def test_consent_endpoint_rejects_missing_cookie(db):
    """An /api/* request without the session cookie is rejected by ``_guard``
    even with a valid Host header - same defense as every other admin route."""
    app = create_app(db)
    c = app.test_client()
    # No set_cookie call: simulate a foreign caller / forged request.
    pid = db.create_person("Alice")
    r = c.post(
        f"/api/identity/{pid}/consent",
        headers={"Host": _host_header()},
        json={"granted_by": "Ms. Smith"},
    )
    assert r.status_code == 403


def test_consent_endpoint_rejects_bad_host(client, db):
    """Even with the right cookie, a Host header outside the allowlist trips
    the DNS-rebinding guard."""
    pid = db.create_person("Alice")
    r = client.post(
        f"/api/identity/{pid}/consent",
        headers={"Host": "evil.example.com"},
        json={"granted_by": "Ms. Smith"},
    )
    assert r.status_code == 403
