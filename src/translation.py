"""Bounded, offline M2M100 inference used by the isolated worker."""

import os
from pathlib import Path
import re

from src.processes import ProcessingError


LANGUAGES = frozenset(('uk', 'pl', 'en', 'ru'))
MAX_TEXT_BYTES = 65_536
MAX_SOURCE_TOKENS = 256
MODEL_FILES = ('model.bin', 'config.json', 'shared_vocabulary.json',
               'vocab.json', 'sentencepiece.bpe.model')
OPTIONAL_MODEL_FILES = ('tokenizer_config.json', 'special_tokens_map.json')


def validate_text(text: str, source: str, target: str) -> None:
    if source not in LANGUAGES or target not in LANGUAGES:
        raise ProcessingError('Translation requires a supported source and target language.')
    if not isinstance(text, str) or not text.strip():
        raise ProcessingError('Translation input is empty.')
    if len(text.encode('utf-8')) > MAX_TEXT_BYTES:
        raise ProcessingError('Translation input is too large.')


def translate_text(text: str, source: str, target: str, tokenizer, translator) -> str:
    """Translate whole sentences; reject truncation instead of returning partial text."""
    validate_text(text, source, target)
    if source == target:
        return text
    tokenizer.src_lang = source
    parts = re.split(r'((?<=[.!?])\s+|\n+)', text.strip())
    chunks = []
    for index in range(0, len(parts), 2):
        if not parts[index].strip():
            continue
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(parts[index], truncation=False))
        if len(tokens) > MAX_SOURCE_TOKENS:
            raise ProcessingError('A sentence exceeds the 256-token translation limit.')
        chunks.append((index, tokens))
    prefix = tokenizer.lang_code_to_token[target]
    output_bytes = sum(len(part.encode('utf-8')) for part in parts[1::2])
    for index, tokens in chunks:
        results = translator.translate_batch(
            [tokens], target_prefix=[[prefix]], beam_size=5, max_input_length=0,
            max_decoding_length=512, end_token=tokenizer.eos_token, return_end_token=True,
        )
        if not results or not results[0].hypotheses:
            raise ProcessingError('Translation returned no result.')
        result = results[0].hypotheses[0]
        if len(result) < 3 or result[0] != prefix or result[-1] != tokenizer.eos_token:
            raise ProcessingError('Translation returned incomplete or invalid output.')
        translated = tokenizer.decode(tokenizer.convert_tokens_to_ids(result[1:-1]),
                                      skip_special_tokens=True).strip()
        if not translated:
            raise ProcessingError('Translation returned empty text.')
        output_bytes += len(translated.encode('utf-8'))
        if output_bytes > MAX_TEXT_BYTES:
            raise ProcessingError('Translation output is too large.')
        parts[index] = translated
    return ''.join(parts)


def load_runtime(model_dir: Path, threads: int = 2):
    if threads < 1:
        raise ProcessingError('Translation threads must be positive.')
    if not all((model_dir / name).is_file() for name in MODEL_FILES):
        raise ProcessingError('Translation model or tokenizer files are missing.')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', USE_TORCH='0', USE_TF='0')
    try:
        import ctranslate2
        from transformers import M2M100Tokenizer

        if 'int8' not in ctranslate2.get_supported_compute_types('cpu'):
            raise ProcessingError('Translation runtime does not support CPU int8.')
        tokenizer = M2M100Tokenizer.from_pretrained(str(model_dir), local_files_only=True)
        translator = ctranslate2.Translator(str(model_dir), device='cpu', compute_type='int8',
                                           inter_threads=1, intra_threads=threads)
        return tokenizer, translator
    except (ImportError, OSError, ValueError, RuntimeError):
        raise ProcessingError('The local translation runtime could not be loaded.') from None


def read_input(path: Path) -> str:
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_TEXT_BYTES + 1)
        if len(raw) > MAX_TEXT_BYTES:
            raise ProcessingError('Translation input is too large.')
        return raw.decode('utf-8')
    except (OSError, UnicodeError):
        raise ProcessingError('Translation input could not be read.') from None
