import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.processes import ProcessingError
from src.transcription import TranscriptionResult


MAX_TRANSCRIPT_BYTES = 65_536
RUNTIME = '/opt/voice-runtime'


def sandbox_command(source: Path, model: Path, language: str, threads: int,
                    timeout: int, structured: bool = False) -> list[str]:
    code = Path(__file__).resolve().parent
    command = [
        '/usr/bin/prlimit', '--fsize=33554432', '--core=0', '--',
        '/usr/bin/bwrap', '--unshare-all', '--unshare-user', '--unshare-net',
        '--unshare-pid', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
        '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
        '--setenv', 'LANG', 'C.UTF-8', '--setenv', 'HOME', '/tmp',
        '--setenv', 'LD_LIBRARY_PATH', '/runtime/lib',
        '--ro-bind', '/usr', '/usr',
        '--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache',
        '--ro-bind', '/etc/alternatives', '/etc/alternatives',
        '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/sbin', '/sbin',
        '--symlink', 'usr/lib', '/lib',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--dir', '/worker', '--dir', '/worker/src',
        '--ro-bind', RUNTIME, '/runtime',
        '--ro-bind', str(source.resolve()), '/input.ogg',
        '--ro-bind', str(model.resolve()), '/model.bin',
    ]
    if Path('/usr/lib64').exists():
        command += ['--symlink', 'usr/lib64', '/lib64']
    for name in ('__init__.py', 'audio_worker.py', 'transcription.py', 'processes.py'):
        command += ['--ro-bind', str(code / name), '/worker/src/' + name]
    command += [
        '--chdir', '/worker', '--', '/usr/bin/python3', '-B', '-m', 'src.audio_worker',
        '--language', language, '--threads', str(threads), '--timeout', str(timeout),
    ]
    if structured:
        command.append('--structured')
    return command


def isolated_transcribe(source: Path, model: Path, *, language: str = 'auto',
                        threads: int = 2, timeout: int = 2400,
                        ffmpeg: str = 'ffmpeg', whisper: str = 'whisper-cli') -> str:
    return _isolated_transcribe(source, model, language=language, threads=threads, timeout=timeout)


def isolated_transcribe_result(source: Path, model: Path, *, language: str = 'auto',
                               threads: int = 2, timeout: int = 2400,
                               ffmpeg: str = 'ffmpeg', whisper: str = 'whisper-cli') -> TranscriptionResult:
    return _isolated_transcribe(source, model, language=language, threads=threads,
                                timeout=timeout, structured=True)


def _isolated_transcribe(source: Path, model: Path, *, language: str,
                         threads: int, timeout: int, structured: bool = False):
    if sys.platform != 'linux':
        raise ProcessingError('Isolated processing requires Linux.')
    if language not in {'auto', 'uk', 'en', 'pl', 'ru'} or threads < 1 or timeout < 1:
        raise ProcessingError('Invalid isolated processing settings.')
    if not source.is_file() or not model.is_file():
        raise ProcessingError('Isolated processing input or model is missing.')
    try:
        with tempfile.TemporaryFile() as output:
            subprocess.run(
                sandbox_command(source, model, language, threads, timeout, structured),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                close_fds=True, check=True, timeout=timeout + 90,
            )
            output.seek(0)
            limit = 400_000 if structured else MAX_TRANSCRIPT_BYTES
            data = output.read(limit + 1)
        if len(data) > limit:
            raise ProcessingError('Isolated transcription output is too large.')
        if structured:
            result = json.loads(data)
            return TranscriptionResult(result['text'], result['language'])
        text = data.decode('utf-8').strip()
        if not text:
            raise ProcessingError('No speech was recognized.')
        return text
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError,
            TypeError, KeyError, RecursionError):
        raise ProcessingError('Isolated processing failed. Check the sandbox setup.') from None
