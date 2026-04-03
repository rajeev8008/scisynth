from __future__ import annotations

import pytest

from scisynth.ingestion.arxiv_single import parse_arxiv_reference


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("http://arxiv.org/abs/1706.03762v7", "1706.03762v7"),
        ("1706.03762", "1706.03762"),
        ("arxiv:1706.03762", "1706.03762"),
    ],
)
def test_parse_arxiv_reference_ok(raw: str, expected: str) -> None:
    assert parse_arxiv_reference(raw) == expected


def test_parse_arxiv_reference_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_arxiv_reference("   ")


def test_parse_arxiv_reference_bad() -> None:
    with pytest.raises(ValueError, match="Could not parse"):
        parse_arxiv_reference("not-a-link")
