"""Offline deployment checks for the optional translation worker."""

import argparse
import os
from pathlib import Path
import sys
import tempfile


def check_inside(host_net: str, host_pid: str, private: Path,
                 runtime: Path, threads: int = 2) -> None:
    if 'BOT_TOKEN' in os.environ or 'ISOLATION_MARKER' in os.environ:
        raise ValueError('Unexpected worker environment.')
    if (os.readlink('/proc/self/ns/net') == host_net
            or os.readlink('/proc/self/ns/pid') == host_pid):
        raise ValueError('Worker namespaces are not private.')
    hidden = (private, Path('/etc/telegram-transcriber.env'), Path('/home'),
              Path('/opt/telegram-transcriber'))
    if any(path.exists() for path in hidden):
        raise ValueError('Host files are visible.')
    if Path('/input.txt').read_bytes() != b'synthetic':
        raise ValueError('Worker input is unavailable.')
    from src.translation import MODEL_FILES, load_runtime
    readonly = [runtime, Path('/usr')] + [Path('/model') / name for name in MODEL_FILES]
    if any(os.access(path, os.W_OK) for path in readonly):
        raise ValueError('Worker resources are writable.')
    tokenizer, translator = load_runtime(Path('/model'), threads)
    if not tokenizer.encode('Hello.', truncation=False) or not translator.model_is_loaded:
        raise ValueError('Worker model could not be initialized.')


def configured_resources():
    runtime = Path(os.environ.get('TRANSLATION_RUNTIME') or '/opt/translation-runtime')
    model = Path(os.environ.get('TRANSLATION_MODEL_DIR') or
                 '/var/lib/translation-models/m2m100-1.2b-int8')
    threads = int(os.environ.get('TRANSLATION_THREADS', '2'))
    if threads < 1:
        raise ValueError('Invalid thread limit.')
    return runtime, model, threads


def check_translation() -> None:
    from src.translation_isolation import (
        run_worker, translation_sandbox_command, validate_translation_installation,
    )
    runtime, model, threads = configured_resources()
    validate_translation_installation(model, runtime)
    with tempfile.TemporaryDirectory(prefix='translation-check-') as directory:
        root = Path(directory)
        source = root / 'input'
        source.write_bytes(b'synthetic')
        private = root / 'private'
        private.write_bytes(b'synthetic')
        command = translation_sandbox_command(source, model, runtime, 'en', 'pl', threads)
        separator = max(i for i, item in enumerate(command) if item == '--')
        command = command[:separator] + [
            '--ro-bind', str(Path(__file__).resolve()), '/worker/verify_translation_isolation.py',
            '--', str(runtime / 'bin/python'), '-B', '/worker/verify_translation_isolation.py',
            '--inside', '--host-net', os.readlink('/proc/self/ns/net'),
            '--host-pid', os.readlink('/proc/self/ns/pid'), '--private-marker', str(private),
            '--runtime', str(runtime), '--threads', str(threads),
        ]
        if run_worker(command, 60).strip() != b'Translation isolation checks passed':
            raise ValueError('Unexpected worker check result.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--host-net')
    parser.add_argument('--host-pid')
    parser.add_argument('--private-marker', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--translate-sample', action='store_true')
    args = parser.parse_args(argv)
    from src.processes import ProcessingError
    try:
        if sys.platform != 'linux':
            raise ValueError('Linux is required.')
        if args.inside:
            if not all((args.host_net, args.host_pid, args.private_marker, args.runtime)):
                raise ValueError('Incomplete worker check arguments.')
            check_inside(args.host_net, args.host_pid, args.private_marker,
                         args.runtime, args.threads)
        else:
            check_translation()
            if args.translate_sample:
                from src.translation_isolation import isolated_translate
                runtime, model, threads = configured_resources()
                result = isolated_translate('Hello. The meeting starts tomorrow at 14:30.',
                                            model, source='en', target='pl', runtime=runtime,
                                            threads=threads, timeout=1200)
                print(result)
        print('Translation isolation checks passed')
        return 0
    except (ProcessingError, OSError, ValueError, RuntimeError):
        print('Translation isolation check failed; keep translation disabled until fixed.',
              file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
