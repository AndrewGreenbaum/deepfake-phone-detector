"""Dataset archive tests use synthetic files only; no download or real dataset."""
import io
import stat
import tarfile
import zipfile

import pytest

from scripts import archive_safety


def tar_fixture(name="dataset/audio.wav", kind=tarfile.REGTYPE):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        good = tarfile.TarInfo("dataset/first.wav")
        good.size = 4
        archive.addfile(good, io.BytesIO(b"good"))
        entry = tarfile.TarInfo(name)
        entry.type = kind
        if kind == tarfile.REGTYPE:
            entry.size = 4
            archive.addfile(entry, io.BytesIO(b"data"))
        else:
            entry.linkname = "../outside.txt"
            archive.addfile(entry)
    buffer.seek(0)
    return tarfile.open(fileobj=buffer)


@pytest.mark.parametrize("name,kind", [
    ("../outside.txt", tarfile.REGTYPE),
    ("/outside.txt", tarfile.REGTYPE),
    ("C:/outside.txt", tarfile.REGTYPE),
    ("..\\outside.txt", tarfile.REGTYPE),
    ("dataset/link", tarfile.SYMTYPE),
    ("dataset/link", tarfile.LNKTYPE),
    ("dataset/fifo", tarfile.FIFOTYPE),
])
def test_tar_rejects_unsafe_archive_before_writing(tmp_path, name, kind):
    destination = tmp_path / "extracted"
    with tar_fixture(name, kind) as archive, pytest.raises(ValueError):
        archive_safety.extract_tar(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_tar_valid_dataset_round_trip(tmp_path):
    with tar_fixture() as archive:
        archive_safety.extract_tar(archive, tmp_path)
    assert (tmp_path / "dataset/audio.wav").read_bytes() == b"data"
    assert (tmp_path / "dataset/first.wav").read_bytes() == b"good"


def test_tar_rejects_preexisting_symlink_escape(tmp_path):
    destination = tmp_path / "extract"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "dataset").symlink_to(outside, target_is_directory=True)
    with tar_fixture() as archive, pytest.raises(ValueError):
        archive_safety.extract_tar(archive, destination)
    assert list(outside.iterdir()) == []


def test_tar_enforces_expanded_size_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(archive_safety, "MAX_TOTAL_BYTES", 6)
    with tar_fixture() as archive, pytest.raises(ValueError, match="limits"):
        archive_safety.extract_tar(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name,mode", [
    ("../outside.txt", stat.S_IFREG),
    ("/outside.txt", stat.S_IFREG),
    ("..\\outside.txt", stat.S_IFREG),
    ("dataset/link", stat.S_IFLNK),
])
def test_zip_rejects_paths_and_links_before_writing(tmp_path, name, mode):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("first.wav", b"good")
        entry = zipfile.ZipInfo(name)
        entry.external_attr = (mode | 0o644) << 16
        archive.writestr(entry, b"data")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        archive_safety.extract_zip(archive, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_zip_selected_audio_round_trip(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/voice_mic1.flac", b"audio")
        archive.writestr("dataset/voice_mic2.flac", b"other")
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        archive_safety.extract_zip(archive, tmp_path, ["dataset/voice_mic1.flac"])
    assert (tmp_path / "dataset/voice_mic1.flac").read_bytes() == b"audio"
    assert not (tmp_path / "dataset/voice_mic2.flac").exists()
