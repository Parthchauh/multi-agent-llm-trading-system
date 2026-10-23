"""Dataset lineage and integrity evidence without assuming point-in-time quality."""

import hashlib
import json
import platform
import shutil
import subprocess  # nosec B404 - fixed read-only Git argv, no shell.
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import pandas as pd

from data.schema import validate_ohlcv


def dataset_manifest(frame, *, provider, query, adjusted, expected_sessions=None):
    frame = validate_ohlcv(frame)
    metadata = {
        "provider": provider,
        "query": query,
        "adjusted": adjusted,
        "timezone": str(frame.index.tz),
        "rows": len(frame),
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
    }
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    metadata["sha256"] = digest.hexdigest()
    metadata["zero_volume_bars"] = int((frame["Volume"] == 0).sum())
    metadata["missing_sessions"] = (
        [stamp.isoformat() for stamp in pd.DatetimeIndex(expected_sessions).difference(frame.index)]
        if expected_sessions is not None
        else None
    )
    metadata["point_in_time_universe_verified"] = False
    metadata["symbol_lifecycle_verified"] = False
    return metadata


def environment_manifest(root):
    root = Path(root).resolve()
    try:
        git_path = shutil.which('git')
        if git_path is None:
            raise FileNotFoundError('Git is unavailable.')
        command = [git_path, "-c", f"safe.directory={root.as_posix()}", '-c', 'core.fsmonitor=false']
        checkout_root = subprocess.check_output(  # nosec B603 - absolute Git path, fixed argv, no shell.
            command + ['rev-parse', '--show-toplevel'], cwd=root,
            stderr=subprocess.DEVNULL, text=True).strip()
        if Path(checkout_root).resolve() != root:
            # An installed wheel below an ancestor checkout is not that checkout.
            raise subprocess.CalledProcessError(1, command)
        commit = subprocess.check_output(  # nosec B603 - fixed read-only Git argv.
            command + ['rev-parse', 'HEAD'],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        source_paths = subprocess.check_output(  # nosec B603 - fixed read-only Git argv.
            command + ["ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        code = hashlib.sha256()
        for relative in sorted(set(source_paths)):
            path = root / relative
            if (
                path.suffix in {".py", ".toml", ".lock", ".html", ".js", ".css", ".yaml", ".yml"}
                and path.is_file()
            ):
                code.update(relative.encode())
                code.update(path.read_bytes())
        source_hash = code.hexdigest()
    except (OSError, subprocess.SubprocessError):
        commit = None
        source_hash = None
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "pydantic", "langchain-core"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    lock = root / "uv.lock"
    installed_hash = None
    try:
        project = distribution('tradingagents')
        digest = hashlib.sha256()
        found = False
        for package_file in sorted(project.files or [], key=str):
            path = Path(str(project.locate_file(package_file)))
            if path.suffix == '.py' and path.is_file():
                digest.update(str(package_file).replace('\\', '/').encode())
                digest.update(path.read_bytes())
                found = True
        if found:
            installed_hash = digest.hexdigest()
    except (PackageNotFoundError, OSError):
        pass
    return {
        "python": sys.version,
        "os": platform.platform(),
        "git_commit": commit,
        "working_tree_code_sha256": source_hash,
        'installed_project_code_sha256': installed_hash,
        "packages": packages,
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "container_digest": None,
    }
