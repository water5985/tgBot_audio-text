import argparse
import sys
from pathlib import Path

from src.processes import ProcessingError
from src.transcription import transcribe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe a local audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--whisper", default="whisper-cli")
    arguments = parser.parse_args(argv)
    try:
        text = transcribe(arguments.audio, arguments.model,
                          ffmpeg=arguments.ffmpeg, whisper=arguments.whisper)
    except ProcessingError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
