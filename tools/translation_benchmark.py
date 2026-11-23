"""Offline M2M100 CPU benchmark; independent of the Telegram bot and its token."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter


LANGUAGES = ('uk', 'en', 'pl', 'ru')
MAX_INPUT_BYTES = 65_536
MAX_SOURCE_TOKENS = 256


class BenchmarkError(Exception):
    """An actionable benchmark error that does not include input text."""


def validate_text(text, source, target):
    if source not in LANGUAGES or target not in LANGUAGES:
        raise BenchmarkError('Source and target must be uk, en, pl or ru.')
    if not text.strip():
        raise BenchmarkError('Input text is empty.')
    if len(text.encode('utf-8')) > MAX_INPUT_BYTES:
        raise BenchmarkError('Input exceeds 65,536 UTF-8 bytes; use a smaller sample.')


def translate_text(text, source, target, tokenizer, translator):
    """Translate sentence-sized chunks, failing rather than truncating long input."""
    validate_text(text, source, target)
    if source == target:
        return text
    tokenizer.src_lang = source
    # Keep separators, including newlines, in their original positions.
    parts = re.split(r'((?<=[.!?])\s+|\n+)', text.strip())
    chunks = []
    for index in range(0, len(parts), 2):
        if not parts[index].strip():
            continue
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(parts[index], truncation=False))
        if len(tokens) > MAX_SOURCE_TOKENS:
            raise BenchmarkError('A sentence exceeds 256 tokens; split it into shorter sentences.')
        chunks.append((index, tokens))
    prefix = tokenizer.lang_code_to_token[target]
    for index, tokens in chunks:
        results = translator.translate_batch(
            [tokens], target_prefix=[[prefix]], beam_size=5,
            max_input_length=0, max_decoding_length=512,
            end_token=tokenizer.eos_token, return_end_token=True,
        )
        if not results or not results[0].hypotheses:
            raise BenchmarkError('The translation model returned no result.')
        result = results[0].hypotheses[0]
        if not result or result[-1] != tokenizer.eos_token:
            raise BenchmarkError('Translation did not finish before its token limit; use a shorter sentence.')
        if result[0] != prefix:
            raise BenchmarkError('The translation model returned an unexpected target language prefix.')
        translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(result[1:-1]),
                                      skip_special_tokens=True).strip()
        if not translated:
            raise BenchmarkError('The translation model returned empty text.')
        parts[index] = translated
    return ''.join(parts)


def load_runtime(model_dir, compute_type, threads):
    if not (model_dir / 'model.bin').is_file():
        raise BenchmarkError('Model directory must contain a converted CTranslate2 model.bin.')
    for filename in ('vocab.json', 'sentencepiece.bpe.model'):
        if not (model_dir / filename).is_file():
            raise BenchmarkError(f'Model directory is missing tokenizer file {filename}; repeat the tokenizer copy step.')
    # Set before imports: inference must never fetch model files or contact the Hub.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    try:
        import ctranslate2
        from transformers import M2M100Tokenizer
    except ImportError:
        raise BenchmarkError('Install translation-requirements.txt in the separate benchmark environment.') from None
    if compute_type not in ctranslate2.get_supported_compute_types('cpu'):
        raise BenchmarkError('This CPU/runtime does not support the requested compute type.')
    tokenizer = M2M100Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
    translator = ctranslate2.Translator(str(model_dir), device='cpu', compute_type=compute_type,
                                       inter_threads=1, intra_threads=threads)
    return tokenizer, translator


def read_input(path):
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise BenchmarkError('Input exceeds 65,536 UTF-8 bytes; use a smaller sample.')
        return raw.decode('utf-8-sig')
    except (OSError, UnicodeError):
        raise BenchmarkError('Cannot read input; provide an existing UTF-8 text file.') from None


def peak_rss_mib():
    if sys.platform != 'linux':
        return None
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--source', choices=LANGUAGES, required=True)
    parser.add_argument('--target', choices=LANGUAGES, required=True)
    parser.add_argument('--compute-type', choices=('float32', 'int8'), default='float32')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.threads < 1:
            raise BenchmarkError('Threads must be a positive integer.')
        text = read_input(args.input)
        validate_text(text, args.source, args.target)
        if args.source == args.target:
            print(text)
            return 0
        started = perf_counter()
        tokenizer, translator = load_runtime(args.model_dir, args.compute_type, args.threads)
        load_seconds = perf_counter() - started
        started = perf_counter()
        first = translate_text(text, args.source, args.target, tokenizer, translator)
        first_seconds = perf_counter() - started
        started = perf_counter()
        warm = translate_text(text, args.source, args.target, tokenizer, translator)
        warm_seconds = perf_counter() - started
        if first != warm:
            raise BenchmarkError('Repeated translations differed; do not use this run for comparison.')
        print(warm)
        print(json.dumps({'load_seconds': round(load_seconds, 3),
                          'first_translation_seconds': round(first_seconds, 3),
                          'warm_translation_seconds': round(warm_seconds, 3),
                          'peak_rss_mib': peak_rss_mib(),
                          'compute_type': translator.compute_type,
                          'threads': args.threads}), file=sys.stderr)
        return 0
    except BenchmarkError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError):
        print('Translation runtime failed. Check the local model, tokenizer and installed dependencies.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
