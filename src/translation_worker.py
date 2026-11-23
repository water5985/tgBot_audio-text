"""One-shot translation worker; only run inside the translation sandbox."""

import argparse
from pathlib import Path
import sys

from src.processes import ProcessingError
from src.translation import LANGUAGES, load_runtime, read_input, translate_text, validate_text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/input.txt'))
    parser.add_argument('--model-dir', type=Path, default=Path('/model'))
    parser.add_argument('--source', choices=sorted(LANGUAGES), required=True)
    parser.add_argument('--target', choices=sorted(LANGUAGES), required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        text = read_input(args.input)
        validate_text(text, args.source, args.target)
        if args.threads < 1:
            return 1
        result = text
        if args.source != args.target:
            tokenizer, translator = load_runtime(args.model_dir, args.threads)
            result = translate_text(text, args.source, args.target, tokenizer, translator)
        sys.stdout.buffer.write(result.encode('utf-8'))
        return 0
    except (ProcessingError, OSError, ValueError, RuntimeError, UnicodeError):
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
