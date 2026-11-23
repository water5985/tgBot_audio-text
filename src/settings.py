import shutil
import sys
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
    audio_sandbox: bool = False
    allow_all_users: bool = False
    translation_enabled: bool = False
    translation_model_dir: Path = Path('/var/lib/translation-models/m2m100-1.2b-int8')
    translation_runtime: Path = Path('/opt/translation-runtime')
    translation_threads: int = 2
    translation_timeout: int = 2400

    def allows(self, user_id: int) -> bool:
        return self.allow_all_users or user_id in self.allowed_user_ids

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
        sandbox = values.get('AUDIO_SANDBOX', 'false').strip().lower()
        if sandbox not in {'true', 'false'}:
            raise ConfigurationError('AUDIO_SANDBOX must be true or false.')
        public = values.get('ALLOW_ALL_USERS', 'false').strip().lower()
        if public not in {'true', 'false'}:
            raise ConfigurationError('ALLOW_ALL_USERS must be true or false.')
        translation = values.get('TRANSLATION_ENABLED', 'false').strip().lower()
        if translation not in {'true', 'false'}:
            raise ConfigurationError('TRANSLATION_ENABLED must be true or false.')
        return cls(
            token=token, allowed_user_ids=allowed,
            model=Path(values.get("WHISPER_MODEL") or "models/ggml-medium.bin").expanduser(),
            whisper=values.get("WHISPER_EXECUTABLE") or "whisper-cli",
            ffmpeg=values.get("FFMPEG_EXECUTABLE") or "ffmpeg",
            language=language,
            threads=positive_integer(values, "WHISPER_THREADS", "2"),
            timeout=positive_integer(values, "TRANSCRIPTION_TIMEOUT", "2400"),
            audio_sandbox=sandbox == 'true',
            allow_all_users=public == 'true',
            translation_enabled=translation == 'true',
            translation_model_dir=Path(values.get('TRANSLATION_MODEL_DIR') or
                                       '/var/lib/translation-models/m2m100-1.2b-int8'),
            translation_runtime=Path(values.get('TRANSLATION_RUNTIME') or '/opt/translation-runtime'),
            translation_threads=positive_integer(values, 'TRANSLATION_THREADS', '2'),
            translation_timeout=positive_integer(values, 'TRANSLATION_TIMEOUT', '2400'),
        )

    def validate_runtime(self) -> None:
        if self.translation_enabled:
            if not self.audio_sandbox:
                raise ConfigurationError('Translation requires AUDIO_SANDBOX=true.')
            from src.processes import ProcessingError
            from src.translation_isolation import validate_translation_installation
            try:
                validate_translation_installation(self.translation_model_dir, self.translation_runtime)
            except ProcessingError:
                raise ConfigurationError('The isolated translation runtime or model is not installed safely.') from None
        if self.audio_sandbox:
            if sys.platform != 'linux' or not all(Path(path).is_file() for path in (
                '/usr/bin/bwrap', '/usr/bin/prlimit', '/usr/bin/ffmpeg',
                '/opt/voice-runtime/bin/whisper-cli',
            )):
                raise ConfigurationError('The Linux audio sandbox runtime is not installed.')
        if not self.allow_all_users and not self.allowed_user_ids:
            return
        if not self.model.is_file():
            raise ConfigurationError("WHISPER_MODEL must point to an existing model file.")
        if self.audio_sandbox:
            return
        for name, executable in [("WHISPER_EXECUTABLE", self.whisper),
                                 ("FFMPEG_EXECUTABLE", self.ffmpeg)]:
            if shutil.which(executable) is None:
                raise ConfigurationError(f"{name} was not found. Set its executable path.")
