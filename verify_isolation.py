"""Linux deployment check; no token values or recordings are printed."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from src.isolation import sandbox_command
from src.processes import ProcessingError


def check_translation() -> None:
    from verify_translation_isolation import check_translation as verify
    verify()


def main() -> int:
    if sys.platform != 'linux' or os.environ.get('AUDIO_SANDBOX', '').lower() != 'true':
        print('Isolation check requires Linux and AUDIO_SANDBOX=true.', file=sys.stderr)
        return 1
    try:
        translation = os.environ.get('TRANSLATION_ENABLED', 'false').strip().lower()
        if translation not in {'true', 'false'}:
            raise ValueError
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.write_bytes(b'synthetic')
            private = root / 'private-marker'
            private.write_text('synthetic', encoding='utf-8')
            host_net = os.readlink('/proc/self/ns/net')
            host_pid = os.readlink('/proc/self/ns/pid')
            check = (
                "import os,pathlib,subprocess; "
                "assert 'BOT_TOKEN' not in os.environ; "
                "assert 'ISOLATION_MARKER' not in os.environ; "
                f"assert os.readlink('/proc/self/ns/net') != {host_net!r}; "
                f"assert os.readlink('/proc/self/ns/pid') != {host_pid!r}; "
                f"assert not pathlib.Path({str(private)!r}).exists(); "
                "assert not pathlib.Path('/etc/telegram-transcriber.env').exists(); "
                "assert not pathlib.Path('/home').exists(); "
                "assert pathlib.Path('/input.ogg').read_bytes() == b'synthetic'; "
                "assert not os.access('/runtime', os.W_OK); "
                "assert not os.access('/usr', os.W_OK); "
                "subprocess.run(['/usr/bin/ffmpeg', '-version'], check=True, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5); "
                "subprocess.run(['/runtime/bin/whisper-cli', '--help'], check=True, "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5); "
                "print('Isolation checks passed')"
            )
            command = sandbox_command(source, source, 'auto', 1, 10)
            command = command[:command.index('/usr/bin/python3')] + [
                '/usr/bin/python3', '-B', '-c', check]
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=20, check=True,
                env={'PATH': '/usr/bin:/bin', 'ISOLATION_MARKER': 'synthetic'},
                close_fds=True,
            )
            if result.stdout.strip() != b'Isolation checks passed':
                raise ValueError
        if translation == 'true':
            check_translation()
        print('Isolation checks passed')
        return 0
    except (OSError, ValueError, ProcessingError, subprocess.SubprocessError):
        print('Isolation check failed; do not disable isolation to bypass this.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
