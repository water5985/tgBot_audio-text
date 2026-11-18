import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class ConfigurationError(Exception):
    pass


def positive_integer(values: Mapping[str, str], name: str, default: str) -> int:
    try:
        value = int(values.get(name, default))
        if value > 0:
            return value
    except ValueError:
        pass
    raise ConfigurationError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class Settings:
    token: str = field(repr=False)
    allowed_user_ids: frozenset[int]
    model: Path
    whisper: str
    ffmpeg: str
    language: str
    threads: int
    timeout: int

    @classmethod
    def from_env(cls, values: Mapping[str, str]) -> "Settings":
        token = values.get("BOT_TOKEN", "").strip()
        if not token or ":" not in token or any(char.isspace() for char in token):
            raise ConfigurationError("Set BOT_TOKEN to the token from BotFather.")
        try:
            allowed = frozenset(int(item.strip()) for item in
                                values.get("ALLOWED_USER_IDS", "").split(",")
                                if item.strip())
            if any(user_id <= 0 for user_id in allowed):
                raise ValueError
        except ValueError:
            raise ConfigurationError("ALLOWED_USER_IDS must contain positive numeric IDs.") from None
        language = values.get("TRANSCRIPTION_LANGUAGE", "auto").strip()
        if language not in {"auto", "en", "pl", "uk", "ru"}:
            raise ConfigurationError("TRANSCRIPTION_LANGUAGE must be auto, en, pl, uk or ru.")
        return cls(
            token=token, allowed_user_ids=allowed,
            model=Path(values.get("WHISPER_MODEL") or "models/ggml-medium.bin").expanduser(),
            whisper=values.get("WHISPER_EXECUTABLE") or "whisper-cli",
            ffmpeg=values.get("FFMPEG_EXECUTABLE") or "ffmpeg",
            language=language,
            threads=positive_integer(values, "WHISPER_THREADS", "2"),
            timeout=positive_integer(values, "TRANSCRIPTION_TIMEOUT", "1200"),
        )

    def validate_runtime(self) -> None:
        if not self.allowed_user_ids:
            return
        if not self.model.is_file():
            raise ConfigurationError("WHISPER_MODEL must point to an existing model file.")
        for name, executable in [("WHISPER_EXECUTABLE", self.whisper),
                                 ("FFMPEG_EXECUTABLE", self.ffmpeg)]:
            if shutil.which(executable) is None:
                raise ConfigurationError(f"{name} was not found. Set its executable path.")
