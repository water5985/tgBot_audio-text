import tempfile
import wave
from pathlib import Path

from src.processes import ProcessingError, run_command


def validate_wave(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            rate = audio.getframerate()
            if frames <= 0 or rate <= 0:
                raise ProcessingError("Audio is empty.")
            if frames > 300 * rate:
                raise ProcessingError("Audio exceeds the five-minute limit.")
    except (OSError, EOFError, wave.Error):
        raise ProcessingError("Audio could not be decoded.") from None


def convert_audio(source: Path, destination: Path, executable: str) -> None:
    run_command([
        executable, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe", "-i", str(source),
        "-map", "0:a:0", "-vn", "-t", "301", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-y", str(destination),
    ], 60, "Audio conversion")
    validate_wave(destination)


def recognize(audio: Path, model: Path, prefix: Path, executable: str,
              language: str, threads: int, timeout: float) -> str:
    run_command([
        executable, "-m", str(model), "-f", str(audio), "-l", language,
        "-t", str(threads), "-otxt", "-of", str(prefix), "-ng",
    ], timeout, "Transcription")
    try:
        text = prefix.with_suffix(".txt").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        raise ProcessingError("Transcription output could not be read.") from None
    if not text:
        raise ProcessingError("No speech was recognized.")
    return text


def transcribe(
    source: Path, model: Path, *, ffmpeg: str = "ffmpeg",
    whisper: str = "whisper-cli", language: str = "auto",
    threads: int = 2, timeout: float = 1200,
) -> str:
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
                             language, threads, timeout)
    except OSError:
        raise ProcessingError("Temporary files could not be managed.") from None
