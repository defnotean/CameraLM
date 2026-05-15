"""Local Flask admin server. Runs in a daemon thread alongside the webcam loop.

Open the admin UI in a browser to manage identities (rename, delete, assign to
classes, drag-and-drop). It is bound to loopback only and protected by a
Host-header allowlist + a SameSite=Strict session cookie, so a malicious website
the user happens to visit cannot reach it (CSRF / DNS-rebinding defense for what
is, after all, a biometric datastore).
"""

import logging
import secrets
from pathlib import Path
from threading import Thread

from flask import Flask, abort, jsonify, make_response, render_template, request, send_file

from .config import ADMIN_HOST, ADMIN_PORT, RETENTION_DAYS
from .identity_db import IdentityDB

log = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_TOKEN = secrets.token_urlsafe(32)          # fresh per process, never persisted
_COOKIE = "cameralm_session"
# Derived from config so changing ADMIN_HOST can't silently desync the allowlist.
_ALLOWED_HOSTS = {
    f"{ADMIN_HOST}:{ADMIN_PORT}",
    f"127.0.0.1:{ADMIN_PORT}",
    f"localhost:{ADMIN_PORT}",
}
_MAX_NAME_LEN = 64


def _clean_name(raw) -> str | None:
    """Validate a person/class name: stripped, non-empty, length-bounded in both chars
    and UTF-8 bytes, no control characters, no zero-width/bidi trickery."""
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    if not name or len(name) > _MAX_NAME_LEN:
        return None
    if len(name.encode("utf-8")) > _MAX_NAME_LEN * 4:
        return None
    for c in name:
        o = ord(c)
        if o < 32 or o == 0x7F:                              # control chars
            return None
        if 0x200B <= o <= 0x200F or 0x202A <= o <= 0x202E or o == 0xFEFF:  # zero-width / bidi
            return None
    return name


def _placeholder_thumbnail() -> Path:
    """Tiny solid-color JPEG used when a person has no real thumbnail yet."""
    from .config import DATA_DIR

    p = DATA_DIR / "placeholder.jpg"
    if not p.exists():
        import cv2
        import numpy as np

        img = np.full((256, 256, 3), 60, dtype="uint8")
        cv2.putText(img, "?", (90, 165), cv2.FONT_HERSHEY_SIMPLEX, 4, (160, 160, 160), 6)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return p


def create_app(db: IdentityDB) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_PACKAGE_ROOT / "templates"),
        # No static assets are served, and an auto-registered /static/ route would
        # sit outside the /api/ guard - disable it entirely.
        static_folder=None,
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # cap request bodies - DoS guard

    @app.before_request
    def _guard():
        # DNS-rebinding defense: a rebound request still carries the attacker's Host.
        if request.host not in _ALLOWED_HOSTS:
            log.warning("Rejected admin request with unexpected Host header: %s", request.host)
            abort(403)
        if not request.path.startswith("/api/"):
            return None
        # CSRF defense: the SameSite=Strict session cookie is not sent on
        # cross-site requests, and a cross-site page cannot read it to forge one.
        if request.cookies.get(_COOKIE) != _TOKEN:
            abort(403)
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            abort(403)
        return None

    @app.errorhandler(400)
    def _bad_request(_e):
        return jsonify({"error": "bad request"}), 400

    @app.errorhandler(403)
    def _forbidden(_e):
        return jsonify({"error": "forbidden"}), 403

    @app.errorhandler(404)
    def _not_found(_e):
        return jsonify({"error": "not found"}), 404

    @app.errorhandler(409)
    def _conflict(_e):
        return jsonify({"error": "conflict"}), 409

    @app.errorhandler(500)
    def _server_error(_e):
        log.exception("Unhandled error in admin request")
        return jsonify({"error": "internal error"}), 500

    @app.route("/")
    def index():
        resp = make_response(render_template("admin.html"))
        # httponly=True: the browser auto-sends the cookie on same-site requests,
        # so the SPA never needs to read it - keep it out of reach of any XSS.
        resp.set_cookie(
            _COOKIE, _TOKEN, samesite="Strict", secure=False, httponly=True, max_age=86400
        )
        return resp

    @app.route("/api/state")
    def state():
        return jsonify(db.snapshot())

    @app.route("/api/identity/<int:pid>/thumbnail")
    def thumbnail(pid):
        if not db.has_person(pid):
            abort(404)
        path = db.thumbnail_path(pid)
        if not path.exists():
            path = _placeholder_thumbnail()
        try:
            return send_file(str(path), mimetype="image/jpeg")
        except OSError:
            abort(404)

    @app.route("/api/identity/<int:pid>", methods=["PATCH"])
    def rename(pid):
        data = request.get_json(silent=True) or {}
        name = _clean_name(data.get("name"))
        if name is None:
            abort(400, "invalid name")
        if not db.rename_person(pid, name):
            abort(404)
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/identity/<int:pid>", methods=["DELETE"])
    def delete_identity(pid):
        if not db.delete_person(pid):
            abort(404)
        db.save()
        return "", 204

    @app.route("/api/identity/<int:pid>/classes", methods=["POST"])
    def add_class(pid):
        data = request.get_json(silent=True) or {}
        name = _clean_name(data.get("name"))
        if name is None:
            abort(400, "invalid class name")
        if not db.add_to_class(pid, name):
            abort(404)
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/identity/<int:pid>/classes/<string:class_name>", methods=["DELETE"])
    def remove_class(pid, class_name):
        if not db.remove_from_class(pid, class_name):
            abort(404)
        db.save()
        return "", 204

    @app.route("/api/class", methods=["POST"])
    def create_class():
        data = request.get_json(silent=True) or {}
        name = _clean_name(data.get("name"))
        if name is None:
            abort(400, "invalid class name")
        if not db.create_class(name):
            abort(409, "class already exists")
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/class/<string:class_name>", methods=["DELETE"])
    def delete_class(class_name):
        if not db.delete_class(class_name):
            abort(404)
        db.save()
        return "", 204

    @app.route("/api/schedule/<string:block>", methods=["PUT"])
    def set_schedule_slot(block):
        """Assign a class to a block, or clear it (empty string).  The schedule
        is per-block - every day shows the same class for a given block, just
        at a different period (the block order rotates by day on the client).
        Unknown class names are rejected so the schedule can't carry a
        dangling reference."""
        data = request.get_json(silent=True) or {}
        raw = data.get("class", "")
        if not isinstance(raw, str):
            abort(400, "class must be a string")
        if not db.set_schedule_slot(block, raw):
            abort(400, "invalid block or unknown class")
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/delete-all", methods=["POST"])
    def delete_all():
        """Privacy control: wipe every identity, embedding, class, and thumbnail.

        Requires an explicit confirmation phrase in the body so a single forged
        or mis-fired POST cannot wipe the entire biometric database.
        """
        data = request.get_json(silent=True) or {}
        if data.get("confirm") != "DELETE ALL":
            abort(400, "confirmation phrase required")
        n = db.delete_all_data()
        db.save()
        log.warning("Admin wiped all identity data (%d people removed).", n)
        return jsonify({"ok": True, "deleted": n})

    @app.route("/api/identity/<int:pid>/consent", methods=["POST"])
    def record_consent(pid):
        """Record affirmative consent for a person.

        Body: {"granted_by": str (required), "notes": str (optional)}.
        granted_by is the operator/role attesting consent - the privacy story
        rests on a real human name being on the record, not a click-through.
        Length-bounded to keep the audit log from being weaponized.
        """
        data = request.get_json(silent=True) or {}
        granted_by_raw = data.get("granted_by", "")
        notes_raw = data.get("notes", "")
        if not isinstance(granted_by_raw, str) or not isinstance(notes_raw, str):
            abort(400, "granted_by and notes must be strings")
        granted_by = granted_by_raw.strip()
        if not granted_by:
            abort(400, "granted_by is required")
        if len(granted_by) > 120 or len(notes_raw) > 500:
            abort(400, "granted_by or notes too long")
        if not db.record_consent(pid, granted_by, notes_raw):
            abort(404)
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/identity/<int:pid>/consent", methods=["DELETE"])
    def revoke_consent(pid):
        """Revoke consent + drop all of that person's biometric vectors.

        Hard break: name and class memberships survive (so the admin UI can
        surface the revoked entry for follow-up), but face/body/partial
        embeddings are gone before this request returns.
        """
        if not db.revoke_consent(pid):
            abort(404)
        db.save()
        return jsonify({"ok": True})

    @app.route("/api/audit")
    def audit_log():
        """Tail the audit log. Query params:
          - ``limit`` (int, default 200, max 1000): cap on entries returned.
          - ``since`` (ISO datetime): return only entries strictly after this.
        Used by the admin UI Privacy panel; also useful for compliance review.
        """
        try:
            limit = int(request.args.get("limit", "200"))
        except ValueError:
            abort(400, "limit must be an integer")
        limit = max(1, min(limit, 1000))
        since = request.args.get("since") or None
        rows = db.read_audit(limit=limit, since=since)
        return jsonify({"entries": rows, "retention_days": RETENTION_DAYS})

    @app.route("/api/purge-stale", methods=["POST"])
    def purge_stale():
        """Run a retention sweep on demand. Drops embeddings for anyone whose
        last_seen_at is older than the configured RETENTION_DAYS. Disabled
        (no-op) when RETENTION_DAYS=0.
        """
        if RETENTION_DAYS <= 0:
            return jsonify({"ok": True, "purged": [], "retention_days": 0})
        purged = db.purge_stale(RETENTION_DAYS)
        if purged:
            db.save()
            log.info("Retention sweep dropped embeddings for %d people.", len(purged))
        return jsonify({"ok": True, "purged": purged, "retention_days": RETENTION_DAYS})

    return app


def start_admin_server(db: IdentityDB, port: int = ADMIN_PORT) -> int:
    app = create_app(db)

    def run():
        try:
            from waitress import serve

            logging.getLogger("waitress").setLevel(logging.WARNING)
            serve(app, host=ADMIN_HOST, port=port, threads=4)
        except ImportError:
            log.info("waitress not installed - using Flask's dev server for the admin UI.")
            app.run(host=ADMIN_HOST, port=port, debug=False, use_reloader=False, threaded=True)
        except Exception:
            log.exception("Admin server crashed.")

    t = Thread(target=run, daemon=True, name="AdminServer")
    t.start()
    return port
