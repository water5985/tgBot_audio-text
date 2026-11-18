import subprocess


class ProcessingError(Exception):
    pass


def run_command(command: list[str], timeout: float, stage: str) -> None:
    try:
        subprocess.run(
            command, check=True, timeout=timeout,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise ProcessingError(f"{stage} timed out.") from None
    except FileNotFoundError:
        raise ProcessingError(f"{stage} executable was not found.") from None
    except (OSError, subprocess.CalledProcessError):
        raise ProcessingError(f"{stage} failed.") from None
