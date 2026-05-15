"""Tests for config.py's post-load invariant validation.

`_validate_config()` runs once at import (so a bad shipped config fails fast).
These confirm the shipped config is self-consistent and that the validator
actually catches a violated invariant rather than passing silently.
"""

import pytest

import cameralm.config as config


def test_shipped_config_passes_validation():
    # If the module imported at all, this already ran clean once; calling it
    # again documents that the shipped values satisfy every invariant.
    config._validate_config()


def test_weak_threshold_above_strong_is_rejected(monkeypatch):
    # The canonical invariant: a weak signal must not be allowed to outrank a
    # strong one. Push FACE_WEAK above FACE_MATCH and the validator must object.
    monkeypatch.setattr(config, "FACE_WEAK_MATCH_THRESH", config.FACE_MATCH_THRESH + 0.05)
    with pytest.raises(ValueError, match="FACE_WEAK_MATCH_THRESH"):
        config._validate_config()


def test_out_of_range_value_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PORT", 70000)
    with pytest.raises(ValueError, match="ADMIN_PORT"):
        config._validate_config()


def test_nonpositive_interval_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "DETECT_EVERY_N", 0)
    with pytest.raises(ValueError, match="DETECT_EVERY_N"):
        config._validate_config()
