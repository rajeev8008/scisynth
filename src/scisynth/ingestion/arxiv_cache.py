from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from scisynth.ingestion.schema import PaperDocument

_MAX = 48


class ArxivPaperCache:
    """Small process-local LRU cache for fetched arXiv PaperDocuments."""

    def __init__(self, max_items: int = _MAX) -> None:
        self._max = max_items
        self._data: OrderedDict[str, PaperDocument] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> PaperDocument | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, doc: PaperDocument) -> None:
        with self._lock:
            self._data[key] = doc
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


arxiv_document_cache = ArxivPaperCache()
