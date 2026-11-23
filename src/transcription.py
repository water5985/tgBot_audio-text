import json
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from src.processes import ProcessingError, run_command


MAX_TRANSCRIPT_BYTES = 65_536
MAX_AUDIO_SECONDS = 600


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None

    def __post_init__(self):
        if (not isinstance(self.text, str) or not self.text.strip()
                or len(self.text.encode('utf-8')) > MAX_TRANSCRIPT_BYTES):
            raise ProcessingError('Transcription output is empty or too large.')
        if self.language is not None and not (
                isinstance(self.language, str) and self.language.isascii()
                and self.language.isalpha() and self.language.islower()
                and len(self.language) in (2, 3)):
            raise ProcessingError('Transcription language metadata is invalid.')


def read_detected_language(path: Path) -> str | None:
    try:
        with path.open('rb') as stream:
            raw = stream.read(1_048_577)
        if len(raw) > 1_048_576:
            return None
        language = json.loads(raw)['result']['language']
        if (isinstance(language, str) and language.isascii() and language.isalpha()
                and language.islower() and len(language) in (2, 3)):
            return language
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError):
        pass
    return None


def validate_wave(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            if frames <= 0 or rate <= 0:
                raise ProcessingError("Audio is empty.")
            if frames > MAX_AUDIO_SECONDS * rate:
                raise ProcessingError("Audio exceeds the 10-minute limit.")
    except (OSError, EOFError, wave.Error):
        raise ProcessingError("Audio could not be decoded.") from None


def convert_audio(source: Path, destination: Path, executable: str) -> None:
    run_command([
        executable, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:a:0", "-vn", "-t", str(MAX_AUDIO_SECONDS + 1), "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-y", str(destination),
    ], 60, "Audio conversion")
    validate_wave(destination)


def recognize(audio: Path, model: Path, prefix: Path, executable: str,
              language: str, threads: int, timeout: float,
              structured: bool = False) -> str | TranscriptionResult:
    run_command([
        executable, "-m", str(model), "-f", str(audio), "-l", language,
        "-t", str(threads), "-otxt", "-of", str(prefix), "-ng",
    ] + (['-oj'] if structured else []), timeout, "Transcription")
    try:
        with prefix.with_suffix('.txt').open('rb') as stream:
            raw = stream.read(MAX_TRANSCRIPT_BYTES + 1)
        if len(raw) > MAX_TRANSCRIPT_BYTES:
            raise ProcessingError('Transcription output is too large.')
        text = raw.decode('utf-8').strip()
    except (OSError, UnicodeError):
        raise ProcessingError("Transcription output could not be read.") from None
    if not text:
        raise ProcessingError("No speech was recognized.")
    if structured:
        detected = language if language != 'auto' else read_detected_language(prefix.with_suffix('.json'))
        return TranscriptionResult(text, detected)
    return text


def transcribe(
    source: Path, model: Path, *, ffmpeg: str = "ffmpeg",
    whisper: str = "whisper-cli", language: str = "auto",
    threads: int = 2, timeout: float = 2400,
) -> str:
    return _transcribe(source, model, ffmpeg=ffmpeg, whisper=whisper,
                       language=language, threads=threads, timeout=timeout)


def transcribe_result(
    source: Path, model: Path, *, ffmpeg: str = 'ffmpeg',
    whisper: str = 'whisper-cli', language: str = 'auto',
    threads: int = 2, timeout: float = 2400,
) -> TranscriptionResult:
    return _transcribe(source, model, ffmpeg=ffmpeg, whisper=whisper,
                       language=language, threads=threads, timeout=timeout, structured=True)


def _transcribe(source: Path, model: Path, *, ffmpeg: str, whisper: str,
                language: str, threads: int, timeout: float,
                structured: bool = False) -> str | TranscriptionResult:
    if not source.is_file():
        raise ProcessingError("Audio file was not found.")
    if not model.is_file():
        raise ProcessingError("Model file was not found.")
    try:
        with tempfile.TemporaryDirectory(prefix="transcription-") as directory:
            workspace = Path(directory)
            audio = workspace / "audio.wav"
            convert_audio(source.resolve(), audio, ffmpeg)
            return recognize(audio, model.resolve(), workspace / "result", whisper,
                             language, threads, timeout, structured)
    except OSError:
        raise ProcessingError("Temporary files could not be managed.") from None
