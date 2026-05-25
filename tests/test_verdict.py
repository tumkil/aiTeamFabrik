"""Tests for the verdict-as-label module."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from factory.core.verdict import (
    ALL_VERDICT_LABELS,
    VERDICT_APPROVE,
    VERDICT_BLOCK,
    VERDICT_CHANGES,
    get_verdict_label,
    is_approved,
    is_changes_requested,
    set_verdict_label,
)


def _mr(labels=None):
    state = {"saved": 0}

    def save():
        state["saved"] += 1

    obj = SimpleNamespace(labels=list(labels or []), save=save)
    obj._state = state
    return obj


class TestSetVerdictLabel:
    def test_apply_approve(self):
        mr = _mr()
        result = set_verdict_label(mr, "green")
        assert result == VERDICT_APPROVE
        assert mr.labels == [VERDICT_APPROVE]
        assert mr._state["saved"] == 1

    def test_apply_changes(self):
        mr = _mr()
        assert set_verdict_label(mr, "yellow") == VERDICT_CHANGES
        assert mr.labels == [VERDICT_CHANGES]

    def test_apply_block(self):
        mr = _mr()
        assert set_verdict_label(mr, "red") == VERDICT_BLOCK
        assert mr.labels == [VERDICT_BLOCK]

    def test_replaces_existing_verdict_label(self):
        mr = _mr(labels=["frontend", VERDICT_BLOCK, "p1"])
        set_verdict_label(mr, "green")
        assert VERDICT_BLOCK not in mr.labels
        assert VERDICT_APPROVE in mr.labels
        # Non-verdict labels preserved.
        assert "frontend" in mr.labels
        assert "p1" in mr.labels

    def test_unknown_color_returns_none(self):
        mr = _mr()
        assert set_verdict_label(mr, "purple") is None
        assert mr.labels == []
        assert mr._state["saved"] == 0

    def test_save_failure_is_swallowed(self, caplog):
        def bad_save():
            raise RuntimeError("network down")

        mr = SimpleNamespace(labels=[], save=bad_save)
        result = set_verdict_label(mr, "green")
        assert result is None
        # Label was still mutated locally; the warning was logged.
        assert any("Failed to save" in r.message for r in caplog.records)


class TestQueries:
    def test_get_verdict_label_none_when_unlabelled(self):
        assert get_verdict_label(_mr()) is None

    def test_get_verdict_label_returns_label(self):
        mr = _mr(labels=["foo", VERDICT_CHANGES])
        assert get_verdict_label(mr) == VERDICT_CHANGES

    @pytest.mark.parametrize("label,expected", [
        (VERDICT_APPROVE, False),
        (VERDICT_CHANGES, True),
        (VERDICT_BLOCK, True),
    ])
    def test_is_changes_requested(self, label, expected):
        assert is_changes_requested(_mr(labels=[label])) is expected

    def test_is_changes_requested_none_when_unlabelled(self):
        assert is_changes_requested(_mr()) is None

    @pytest.mark.parametrize("label,expected", [
        (VERDICT_APPROVE, True),
        (VERDICT_CHANGES, False),
        (VERDICT_BLOCK, False),
    ])
    def test_is_approved(self, label, expected):
        assert is_approved(_mr(labels=[label])) is expected

    def test_is_approved_none_when_unlabelled(self):
        assert is_approved(_mr()) is None


def test_all_verdict_labels_inventory():
    assert ALL_VERDICT_LABELS == {VERDICT_APPROVE, VERDICT_CHANGES, VERDICT_BLOCK}
