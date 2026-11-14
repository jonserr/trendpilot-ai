"""Project-only external API-key store. Never print or accept key values in argv.

Usage: python3 scripts/api_key.py {init,status,rotate,run -- COMMAND ...}
"""

import argparse
import os
from pathlib import Path
import secrets
import stat
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def key_path() -> Path:
    path = Path(
        os.environ.get("TRENDPILOT_API_KEY_FILE")
        or Path.home() / ".config" / "trendpilot-ai" / "api-key"
    ).expanduser()
    if not path.is_absolute():
        raise ValueError("TRENDPILOT_API_KEY_FILE must be an absolute path outside the project")
    if path.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError("API-key storage must be outside the project directory")
    return path


def check_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("Key directory must be a real directory owned by the current user")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("Key directory permissions must be 0700")


def check_key(path: Path) -> None:
    """Reject symlinks, shared files, permissive modes, and invalid stored values."""
    check_directory(path.parent)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise ValueError("Key must be a regular, unshared file owned by the current user")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ValueError("Key file permissions must be 0600")
        raw = handle.read(4099)
        if len(raw) > 4098:
            raise ValueError("Stored API-key file is too large")
        value = raw.rstrip(b"\r\n")
        if not value or len(value) > 4096 or any(not 33 <= char <= 126 for char in value):
            raise ValueError("Stored API key must be nonempty printable ASCII without whitespace")


def initialize(path: Path) -> bool:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        check_key(path)
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(secrets.token_urlsafe(32) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink()
        raise
    return True


def rotate(path: Path) -> None:
    check_key(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(secrets.token_urlsafe(32) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["init", "status", "rotate", "run"])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.action == "run" and not command:
        parser.error("run requires -- COMMAND [ARGS ...]")
    if args.action != "run" and command:
        parser.error("only run accepts a command")
    try:
        path = key_path()
        if args.action == "init":
            created = initialize(path)
            print(f"API key {'created' if created else 'already configured'}: {path}")
        elif args.action == "rotate":
            rotate(path)
            print(f"API key rotated: {path}. Recreate API and dashboard containers together.")
        elif args.action == "status":
            check_key(path)
            print(f"API key configured: {path} (directory 0700, file 0600)")
        else:
            check_key(path)
            env = os.environ.copy()
            env.pop("API_KEY", None)
            env.update(
                API_KEY_FILE=str(path), API_KEY_REQUIRED="true", TRENDPILOT_API_KEY_FILE=str(path)
            )
            return subprocess.run(command, env=env, check=False).returncode
    except ValueError as exc:
        parser.exit(1, f"{exc}\n")
    except OSError:
        parser.exit(1, "Cannot access key store or command. Run init and check file permissions.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
