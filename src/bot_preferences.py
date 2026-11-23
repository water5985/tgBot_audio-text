"""Immutable choices captured when a recording is accepted."""

from dataclasses import dataclass


LANGUAGES = {'uk': 'Українська', 'en': 'English', 'pl': 'Polski',
             'ru': 'Русский', 'auto': 'Automatic detection'}


@dataclass(frozen=True)
class JobOptions:
    mode: str = 'transcribe'
    source: str = 'auto'
    target: str | None = None
