"""
event_dispatcher.py - Forward events to an external backend via HTTP POST.

Optional component: only used when BACKEND_WEBHOOK_URL is configured.
Falls back gracefully to logging-only mode when no URL is set.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict

import httpx

logger = logging.getLogger(__name__)


class EventDispatcher:
    """
    Non-blocking HTTP dispatcher with retry and backoff.

    Usage
    -----
    dispatcher = EventDispatcher()
    await dispatcher.dispatch(event_dict)
    """

    def __init__(
        self,
        backend_url: str = "",
        timeout_seconds: float = 2.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
    ) -> None:
        self.backend_url = backend_url or os.getenv("BACKEND_WEBHOOK_URL", "")
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def dispatch(self, event: Dict) -> bool:
        """
        Send event to the configured backend URL.

        Returns True on success, False if all attempts fail.
        If no URL is configured, logs the event and returns False.
        """
        if not self.backend_url:
            logger.debug("No backend URL configured. Event: %s", event)
            return False

        for attempt in range(1, self.max_retries + 2):
            try:
                response = await self._client.post(
                    self.backend_url,
                    json=event,
                )
                if response.is_success:
                    return True
                logger.warning(
                    "Dispatch attempt %d failed: HTTP %d",
                    attempt, response.status_code,
                )
            except httpx.RequestError as exc:
                logger.warning("Dispatch attempt %d error: %s", attempt, exc)

            if attempt <= self.max_retries:
                await asyncio.sleep(self.retry_delay)

        logger.error("All dispatch attempts failed for event: %s", event.get("event_type"))
        return False

    async def close(self) -> None:
        await self._client.aclose()
