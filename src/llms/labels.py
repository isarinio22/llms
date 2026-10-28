"""The label set and strict validation of model outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

_LABEL_LINE = re.compile(r"label\s*[:=]\s*(.+)", re.IGNORECASE)


def normalize_label(text: str) -> str:
    """Canonical form for matching: lowercase, spaces/hyphens -> underscores, no wrapping punctuation."""
    s = text.strip().strip("`'\"*.?!,;: \t")
    s = re.sub(r"[\s\-]+", "_", s.lower())
    return s


@dataclass(frozen=True)
class LabelSet:
    names: tuple[str, ...]
    descriptions: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @cached_property
    def _lookup(self) -> dict[str, int]:
        return {normalize_label(n): i for i, n in enumerate(self.names)}

    def parse(self, raw: str | None) -> int | None:
        """Return the label id if `raw` is exactly one valid label (after normalization), else None.

        Accepts a bare label or a 'Label: <name>' line (rationale format). Anything else is invalid:
        no fuzzy matching, so the invalid rate is an honest measure of instruction following.
        """
        if raw is None:
            return None
        text = raw.strip()
        match = _LABEL_LINE.search(text)
        if match:
            text = match.group(1)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) != 1:
            return None
        return self._lookup.get(normalize_label(lines[0]))


def load_labels(path: str | Path) -> LabelSet:
    items = yaml.safe_load(Path(path).read_text())["labels"]
    return LabelSet(
        names=tuple(it["name"] for it in items),
        descriptions=tuple(it["description"] for it in items),
    )
