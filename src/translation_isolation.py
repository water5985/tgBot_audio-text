"""Launch a one-shot translator without bot credentials or network access."""

import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import tempfile

from src.processes import ProcessingError
from src.translation import MAX_TEXT_BYTES, MODEL_FILES, OPTIONAL_MODEL_FILES, validate_text


def validate_resource_locations(model_dir, runtime) -> None:
    model = PurePosixPath(str(model_dir))
    environment = PurePosixPath(str(runtime))
    if (model.parent != PurePosixPath('/var/lib/translation-models')
            or model.name in {'', '.', '..'}
            or environment.parent != PurePosixPath('/opt')
            or not (environment.name == 'translation-runtime'
                    or environment.name.startswith('translation-runtime-'))):
        raise ProcessingError('Translation resources must use dedicated installation directories.')


def require_root_owned(path) -> None:
    details = path.stat()
    if details.st_uid != 0 or details.st_mode & 0o022:
        raise ProcessingError('Translation resources must be root-owned and not writable by other users.')


def validate_runtime_contents(runtime: Path) -> None:
    for path in runtime.rglob('*'):
        if path.name == '.env' or path.name.startswith('.env.') or path.name == 'telegram-transcriber.env':
            raise ProcessingError('Translation runtime must not contain bot configuration files.')
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not (resolved.is_relative_to(runtime) or resolved.is_relative_to('/usr')):
                raise ProcessingError('Translation runtime contains an unsafe symlink.')
        require_root_owned(path)


def validate_translation_installation(model_dir: Path, runtime: Path) -> None:
    """Fail closed on broad mounts, writable runtime code or redirected resources."""
    if sys.platform != 'linux':
        raise ProcessingError('Isolated translation requires Linux.')
    validate_resource_locations(model_dir, runtime)
    try:
        for directory in (model_dir, runtime):
            if not directory.is_dir() or directory.resolve(strict=True) != directory:
                raise ProcessingError('Translation installation directories must not be symlinks.')
            for ancestor in (directory, *directory.parents):
                require_root_owned(ancestor)
        for name in MODEL_FILES + OPTIONAL_MODEL_FILES:
            path = model_dir / name
            if name in OPTIONAL_MODEL_FILES and not path.exists():
                continue
            if not path.is_file() or path.is_symlink():
                raise ProcessingError('Translation model files are missing or redirected.')
            require_root_owned(path)
        if not (runtime / 'bin/python').is_file() or not (runtime / 'pyvenv.cfg').is_file():
            raise ProcessingError('The dedicated translation environment is missing.')
        validate_runtime_contents(runtime)
        if not all(Path(path).is_file() for path in ('/usr/bin/bwrap', '/usr/bin/prlimit')):
            raise ProcessingError('Translation sandbox tools are missing.')
    except (OSError, ValueError, RuntimeError):
        raise ProcessingError('The isolated translation installation is unavailable.') from None


def translation_sandbox_command(input_path: Path, model_dir: Path, runtime: Path,
                                source: str, target: str, threads: int) -> list[str]:
    code = Path(__file__).resolve().parent
    command = [
        '/usr/bin/prlimit', '--fsize=65537', '--core=0', '--',
        '/usr/bin/bwrap', '--unshare-all', '--unshare-user', '--unshare-net',
        '--unshare-pid', '--die-with-parent', '--new-session', '--cap-drop', 'ALL',
        '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
        '--setenv', 'LANG', 'C.UTF-8', '--setenv', 'HOME', '/tmp',
        '--setenv', 'HF_HUB_OFFLINE', '1', '--setenv', 'TRANSFORMERS_OFFLINE', '1',
        '--setenv', 'HF_HUB_DISABLE_TELEMETRY', '1', '--setenv', 'USE_TORCH', '0',
        '--setenv', 'USE_TF', '0',
        '--ro-bind', '/usr', '/usr',
        '--ro-bind', '/etc/ld.so.cache', '/etc/ld.so.cache',
        '--ro-bind', '/etc/alternatives', '/etc/alternatives',
        '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/sbin', '/sbin',
        '--symlink', 'usr/lib', '/lib',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--dir', '/worker', '--dir', '/worker/src', '--dir', '/model',
        '--ro-bind', str(runtime), str(runtime),
        '--ro-bind', str(input_path.resolve()), '/input.txt',
    ]
    if Path('/usr/lib64').exists():
        command += ['--symlink', 'usr/lib64', '/lib64']
    for name in MODEL_FILES + OPTIONAL_MODEL_FILES:
        if name in MODEL_FILES or (model_dir / name).is_file():
            command += ['--ro-bind', str(model_dir / name), '/model/' + name]
    for name in ('__init__.py', 'translation.py', 'translation_worker.py', 'processes.py'):
        command += ['--ro-bind', str(code / name), '/worker/src/' + name]
    command += [
        '--chdir', '/worker', '--', str(runtime / 'bin/python'), '-B', '-m', 'src.translation_worker',
        '--input', '/input.txt', '--model-dir', '/model', '--source', source,
        '--target', target, '--threads', str(threads),
    ]
    return command


def run_worker(command: list[str], timeout: int) -> bytes:
    """Reap a timed-out sandbox before returning to the serialized queue."""
    try:
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'},
                close_fds=True, start_new_session=True,
            )
            try:
                status = process.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    process.wait()
                raise
            if status != 0:
                raise ProcessingError('Isolated translation failed.')
            output.seek(0)
            return output.read(MAX_TEXT_BYTES + 1)
    except (OSError, subprocess.SubprocessError):
        raise ProcessingError('Isolated translation failed or timed out.') from None


def isolated_translate(text: str, model_dir: Path, *, source: str, target: str,
                       runtime: Path, threads: int = 2, timeout: int = 2400) -> str:
    validate_text(text, source, target)
    if sys.platform != 'linux' or threads < 1 or timeout < 1:
        raise ProcessingError('Isolated translation requires Linux and positive limits.')
    if source == target:
        return text
    validate_translation_installation(model_dir, runtime)
    try:
        with tempfile.TemporaryDirectory(prefix='translation-') as directory:
            input_path = Path(directory) / 'input.txt'
            input_path.write_text(text, encoding='utf-8')
            command = translation_sandbox_command(input_path, model_dir, runtime, source, target, threads)
            raw = run_worker(command, timeout)
        if len(raw) > MAX_TEXT_BYTES:
            raise ProcessingError('Translation output is too large.')
        result = raw.decode('utf-8').strip()
        if not result:
            raise ProcessingError('Translation returned empty output.')
        return result
    except (OSError, UnicodeError):
        raise ProcessingError('Isolated translation input or output is invalid.') from None
