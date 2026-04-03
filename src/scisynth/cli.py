from __future__ import annotations

import logging


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    from scisynth.config import get_settings

    s = get_settings()
    uvicorn.run(
        "scisynth.api.main:app",
        host=s.api_host,
        port=s.api_port,
        factory=False,
    )
