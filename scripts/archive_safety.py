"""Extract dataset archives as data, never as filesystem instructions."""

from pathlib import Path, PurePosixPath
import stat

MAX_MEMBERS = 1_000_000
MAX_MEMBER_BYTES = 2 * 1024 ** 3
MAX_TOTAL_BYTES = 64 * 1024 ** 3


def _destination(root, name):
    parts = PurePosixPath(name)
    if (not name or parts.is_absolute() or ".." in parts.parts
            or "\\" in name or ":" in name or "\x00" in name):
        raise ValueError("Archive contains an unsafe path")
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Archive path escapes the dataset folder")


def _check_size(count, size, total):
    if count > MAX_MEMBERS or size < 0 or size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
        raise ValueError("Archive exceeds dataset extraction limits")


def extract_tar(archive, destination):
    """Preflight every entry before writing; datasets may contain files/directories only."""
    root = Path(destination).resolve()
    members, total = [], 0
    for member in archive:
        _destination(root, member.name)
        if not (member.isfile() or member.isdir()):
            raise ValueError("Archive links and special files are not allowed")
        total += member.size
        _check_size(len(members) + 1, member.size, total)
        members.append(member)
    archive.extractall(root, members=members, filter="data")


def extract_zip(archive, destination, members=None):
    """Use the same path/type/size policy for ZIP dataset preparation."""
    root = Path(destination).resolve()
    selected = archive.infolist() if members is None else [archive.getinfo(name) for name in members]
    total = 0
    for count, member in enumerate(selected, 1):
        _destination(root, member.filename)
        kind = stat.S_IFMT(member.external_attr >> 16)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Archive links and special files are not allowed")
        total += member.file_size
        _check_size(count, member.file_size, total)
    archive.extractall(root, members=selected)
