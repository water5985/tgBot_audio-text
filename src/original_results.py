"""Short-lived, private, memory-only originals for translation replies."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import secrets


@dataclass(frozen=True)
class Original:
    user_id: int
    chat_id: int
    text: str
    expires: float
    timer: asyncio.TimerHandle


class OriginalResults:
    def __init__(self, ttl: float = 600):
        self.ttl = ttl
        self._items: OrderedDict[str, Original] = OrderedDict()

    def __len__(self):
        return len(self._items)

    def _remove(self, key):
        item = self._items.pop(key, None)
        if item is not None:
            item.timer.cancel()

    def put(self, user_id: int, chat_id: int, text: str) -> str:
        if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > 65_536:
            raise ValueError('Invalid original transcript.')
        while len(self._items) >= 32:
            self._remove(next(iter(self._items)))
        key = secrets.token_urlsafe(18)
        loop = asyncio.get_running_loop()
        timer = loop.call_later(self.ttl, self._remove, key)
        self._items[key] = Original(user_id, chat_id, text, loop.time() + self.ttl, timer)
        return key

    def get(self, key: str, user_id: int, chat_id: int) -> str | None:
        item = self._items.get(key)
        if item is None:
            return None
        if asyncio.get_running_loop().time() >= item.expires:
            self._remove(key)
            return None
        if (item.user_id, item.chat_id) != (user_id, chat_id):
            return None
        return item.text

    def close(self):
        for key in list(self._items):
            self._remove(key)
