import argparse
import json
import sys
from pathlib import Path

from src.processes import ProcessingError
from src.transcription import transcribe, transcribe_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['auto', 'uk', 'en', 'pl', 'ru'], required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--timeout', type=int, required=True)
    parser.add_argument('--structured', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or args.timeout < 1:
        return 1
    try:
        recognize = transcribe_result if args.structured else transcribe
        result = recognize(
            Path('/input.ogg'), Path('/model.bin'), ffmpeg='/usr/bin/ffmpeg',
            whisper='/runtime/bin/whisper-cli', language=args.language,
            threads=args.threads, timeout=args.timeout,
        )
        text = (json.dumps({'text': result.text, 'language': result.language}, ensure_ascii=False)
                if args.structured else result)
        sys.stdout.buffer.write(text.encode('utf-8'))
        return 0
    except ProcessingError:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
