import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest

from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.truncate_output import truncate_output


class FakeInterpreter:
    """The spill bookkeeping on its own, without building a whole interpreter."""

    _spill_file_path = OpenInterpreter._spill_file_path
    _open_spill = OpenInterpreter._open_spill
    _record_full_output = OpenInterpreter._record_full_output

    def __init__(self, max_output=100):
        self.max_output = max_output
        self._spill_path = None
        self._spill_message = None
        self._spill_index = 0
        self._spill_pending = ""
        self._spill_size = 0
        self._spill_body_end = None


def feed(oi, message, text, chunk_size=10):
    """Stream `text` into the archive the way console chunks arrive."""
    note = None
    for i in range(0, len(text), chunk_size):
        note = oi._record_full_output(message, text[i : i + chunk_size])
    return note


@pytest.fixture
def spill_dir(tmp_path, monkeypatch):
    """Point the shared temp directory at tmp_path, where the archive is created."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return tmp_path


def test_nothing_is_written_while_the_output_fits(spill_dir):
    oi = FakeInterpreter(max_output=100)
    note = feed(oi, {}, "x" * 50)
    assert note is None
    assert oi._spill_path is None
    assert not any(spill_dir.iterdir())


def test_full_output_is_recoverable_after_truncation(spill_dir):
    oi = FakeInterpreter(max_output=100)
    payload = "".join(f"line {i}\n" for i in range(500))
    note = feed(oi, {}, payload)

    assert note is not None
    assert payload in Path(oi._spill_path).read_text()


def test_each_block_is_appended_not_overwritten(spill_dir):
    oi = FakeInterpreter(max_output=100)
    first = "A" * 400
    second = "B" * 400
    feed(oi, {"m": 1}, first)
    feed(oi, {"m": 2}, second)

    contents = Path(oi._spill_path).read_text()
    assert first in contents
    assert second in contents
    assert contents.index(first) < contents.index(second)
    assert "OI OUTPUT BLOCK 1" in contents
    assert "OI OUTPUT BLOCK 2" in contents


def test_the_awk_range_in_the_note_extracts_exactly_that_block(spill_dir):
    oi = FakeInterpreter(max_output=100)
    feed(oi, {"m": 1}, "AAAA\n" * 100)
    feed(oi, {"m": 2}, "BBBB\n" * 100)

    extracted = subprocess.run(
        ["awk", "/^===== OI OUTPUT BLOCK 2 =/,/^===== END BLOCK 2 /", oi._spill_path],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "BBBB" in extracted
    assert "AAAA" not in extracted


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("awk") is None,
    reason="needs bash and awk",
)
@pytest.mark.parametrize("dirname", ["Jane Doe", "Local\\Temp"])
def test_the_awk_command_in_the_note_runs_as_given(spill_dir, monkeypatch, dirname):
    """The awk command in the note must work when pasted into bash verbatim.

    The model runs it through the bash tool (GNU bash everywhere, Git Bash on
    Windows), so the archive path needs POSIX quoting. Unquoted, a temp dir
    with a space (`C:\\Users\\Jane Doe\\...`) splits into two arguments, and
    bash strips the backslashes out of any Windows path, so awk is handed a
    file that does not exist and the cut output is unreachable.
    """
    temp_dir = spill_dir / dirname
    temp_dir.mkdir(parents=True)
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    oi = FakeInterpreter(max_output=100)
    note = feed(oi, {}, "CCCC\n" * 100)

    command = re.search(r"`(awk .*)`", note).group(1)
    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, timeout=30
    )
    assert "CCCC" in result.stdout, result.stderr


def test_archive_is_not_world_readable(spill_dir):
    oi = FakeInterpreter(max_output=100)
    feed(oi, {}, "x" * 400)

    mode = stat.S_IMODE(os.stat(oi._spill_path).st_mode)
    assert not mode & stat.S_IRGRP
    assert not mode & stat.S_IROTH


def test_archive_never_writes_into_a_file_planted_at_its_path(spill_dir):
    """The archive must be a file this interpreter created, never one it found.

    The temp directory is shared, so another local user can pre-create a file
    at any predictable name - here the old `oi_outputs_<pid>.log`, made
    world-writable. Opening whatever already exists at that path would append
    the full untruncated output (credentials included) to the attacker's file.
    The archive has to be created exclusively at a name nobody can guess.
    """
    planted = spill_dir / f"oi_outputs_{os.getpid()}.log"
    planted.write_text("PLANTED\n")
    planted.chmod(0o666)

    oi = FakeInterpreter(max_output=100)
    payload = "secret-output " * 50
    note = feed(oi, {}, payload)

    assert planted.read_text() == "PLANTED\n"
    assert note is not None
    assert payload in Path(oi._spill_path).read_text()


def test_each_interpreter_gets_its_own_archive(spill_dir):
    """Two interpreters in one process must not share an archive file.

    Each keeps its own block numbering and its own resume offset. Sharing one
    file (as a pid-based name does, and as a later process reusing the pid
    would) means one interpreter resumes at its saved offset, overwrites the
    other's block and truncates the rest away, and both write a "BLOCK 1" that
    the awk range in the note would then print together.
    """
    a = FakeInterpreter(max_output=100)
    b = FakeInterpreter(max_output=100)
    message_a = {}
    feed(a, message_a, "A" * 400)
    feed(b, {}, "B" * 400)
    feed(a, message_a, "a" * 400)  # a resumes its block 1

    b_contents = Path(b._spill_path).read_text()
    assert "B" * 400 in b_contents
    assert b_contents.count("OI OUTPUT BLOCK 1 ") == 1
    assert "A" * 400 + "a" * 400 in Path(a._spill_path).read_text()
    assert a._spill_path != b._spill_path


def test_unusable_temp_dir_still_truncates_without_an_archive(spill_dir, monkeypatch):
    """If the archive cannot be created, recording must not raise.

    Truncation has to keep working on a read-only or missing temp directory;
    the only loss is the archive, signalled by returning no note.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(spill_dir / "does-not-exist"))
    oi = FakeInterpreter(max_output=100)

    assert feed(oi, {}, "x" * 400) is None


def test_writes_stay_linear_in_the_size_of_the_output(spill_dir):
    """Each chunk must append only what is new.

    Rewriting the whole block per chunk is O(n^2): output arrives per line, so a
    few hundred KB over a few thousand lines turns into hundreds of MB written.
    """
    oi = FakeInterpreter(max_output=100)

    written = 0
    real_open_spill = oi._open_spill

    class CountingFile:
        def __init__(self, f):
            self._f = f

        def write(self, data):
            nonlocal written
            written += len(data)
            return self._f.write(data)

        def __getattr__(self, name):
            return getattr(self._f, name)

        def __enter__(self):
            self._f.__enter__()
            return self

        def __exit__(self, *exc):
            return self._f.__exit__(*exc)

    oi._open_spill = lambda p: CountingFile(real_open_spill(p))

    payload = "y" * 20_000
    feed(oi, {}, payload, chunk_size=20)  # 1000 chunks

    # Body written once, plus one small footer rewrite per chunk.
    assert written < 4 * len(payload)


def test_truncate_output_leads_with_shaping_advice_and_ends_with_the_note():
    out = truncate_output("z" * 5000, max_output_chars=100, spill_note="ARCHIVE_NOTE.")
    assert "echo OK" in out
    assert "FAILED" in out
    assert "ARCHIVE_NOTE." in out
    assert out.index("echo OK") < out.index("ARCHIVE_NOTE.")


def test_truncate_output_without_an_archive_still_advises():
    out = truncate_output("z" * 5000, max_output_chars=100)
    assert "grep" in out
    assert "archived at" not in out


@pytest.mark.parametrize("max_output", [50, 100, 1000])
def test_short_output_is_left_alone(max_output):
    data = "small"
    assert truncate_output(data, max_output_chars=max_output) == data
