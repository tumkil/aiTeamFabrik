"""Verdict-as-label helpers for SoftwareTeamFabrik.

The monitor needs to know the *current* review verdict on every MR on every
poll. Historically it parsed the rendered markdown of the latest factory
review note with regex on substrings like ``"REQUEST CHANGES"`` and
``"APPROVE"``. That coupling is fragile — a meta-reviewer that quotes the
primary review, an emoji variation selector, or a future change to the
comment template can all misclassify.

GitLab labels are O(1), structured, and survive comment-template churn.
This module is the single source of truth for the three verdict labels and
the helpers to read/write them.

Comment scanning remains the *fallback* in monitor for older MRs reviewed
before this module existed; new reviews always set the label.
"""

from __future__ import annotations

from typing import Optional

import logging

logger = logging.getLogger(__name__)

# The three verdict labels, in increasing severity.
VERDICT_APPROVE = "factory-verdict-approve"
VERDICT_CHANGES = "factory-verdict-changes"
VERDICT_BLOCK = "factory-verdict-block"

ALL_VERDICT_LABELS: frozenset[str] = frozenset({
    VERDICT_APPROVE,
    VERDICT_CHANGES,
    VERDICT_BLOCK,
})

# Map verdict-color (the existing internal vocabulary used by review_mr)
# to the corresponding label.
_COLOR_TO_LABEL: dict[str, str] = {
    "green": VERDICT_APPROVE,
    "yellow": VERDICT_CHANGES,
    "red": VERDICT_BLOCK,
}


def set_verdict_label(merge_request, color: str) -> Optional[str]:
    """Apply the verdict label that matches *color* to *merge_request*.

    Strips any previously-applied factory verdict labels first so the MR
    always carries exactly one verdict at a time. Best-effort: a network
    failure on save is logged, never raised — the comment was already posted
    and is still authoritative.

    Parameters
    ----------
    merge_request:
        A python-gitlab ``ProjectMergeRequest`` (or any object with
        ``labels: list[str]`` and ``save()``).
    color:
        One of ``"green"`` / ``"yellow"`` / ``"red"`` — the same vocabulary
        used by ``factory.commands.review_mr._verdict_color``.

    Returns
    -------
    The label that was applied, or ``None`` if *color* was unknown.
    """
    label = _COLOR_TO_LABEL.get(color)
    if label is None:
        logger.warning("Unknown verdict color %r; not labelling MR", color)
        return None

    current = list(getattr(merge_request, "labels", []) or [])
    cleaned = [lbl for lbl in current if lbl not in ALL_VERDICT_LABELS]
    cleaned.append(label)
    merge_request.labels = cleaned

    try:
        merge_request.save()
    except Exception as exc:  # noqa: BLE001 — best-effort metadata update
        logger.warning("Failed to save verdict label %s on MR: %s", label, exc)
        return None
    return label


def get_verdict_label(merge_request) -> Optional[str]:
    """Return the currently-applied factory verdict label, or ``None``."""
    for lbl in getattr(merge_request, "labels", []) or []:
        if lbl in ALL_VERDICT_LABELS:
            return lbl
    return None


def is_changes_requested(merge_request) -> Optional[bool]:
    """Return True/False from the verdict label, or ``None`` if unlabelled.

    ``None`` is meaningful: it means the caller cannot determine the verdict
    from labels alone and should fall back to comment scanning.
    """
    lbl = get_verdict_label(merge_request)
    if lbl is None:
        return None
    return lbl in (VERDICT_CHANGES, VERDICT_BLOCK)


def is_approved(merge_request) -> Optional[bool]:
    """Return True/False from the verdict label, or ``None`` if unlabelled."""
    lbl = get_verdict_label(merge_request)
    if lbl is None:
        return None
    return lbl == VERDICT_APPROVE
