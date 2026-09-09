import os
import stat
import subprocess

import pytest

from interpreter.core.core import OpenInterpreter
from interpreter.core.utils.truncate_output import truncate_output


class FakeInterpreter:
    """The spill bookkeeping on its own, without building a whole interpreter."""

    _spill_file_path = OpenInterpreter._spill_file_path
    _open_spill = OpenInterpreter._open_spill
    _record_full_output = OpenInterpreter._record_full_output

    def __init__(self, path, max_output=100):
        self.max_output = max_output
        self._spill_path = str(path)
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


def test_nothing_is_written_while_the_output_fits(tmp_path):
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)
    note = feed(oi, {}, "x" * 50)
    assert note is None
    assert not path.exists()


def test_full_output_is_recoverable_after_truncation(tmp_path):
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)
    payload = "".join(f"line {i}\n" for i in range(500))
    note = feed(oi, {}, payload)

    assert note is not None
    assert payload in path.read_text()


def test_each_block_is_appended_not_overwritten(tmp_path):
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)
    first = "A" * 400
    second = "B" * 400
    feed(oi, {"m": 1}, first)
    feed(oi, {"m": 2}, second)

    contents = path.read_text()
    assert first in contents
    assert second in contents
    assert contents.index(first) < contents.index(second)
    assert "OI OUTPUT BLOCK 1" in contents
    assert "OI OUTPUT BLOCK 2" in contents


def test_the_awk_range_in_the_note_extracts_exactly_that_block(tmp_path):
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)
    feed(oi, {"m": 1}, "AAAA\n" * 100)
    feed(oi, {"m": 2}, "BBBB\n" * 100)

    extracted = subprocess.run(
        ["awk", "/^===== OI OUTPUT BLOCK 2 =/,/^===== END BLOCK 2 /", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    assert "BBBB" in extracted
    assert "AAAA" not in extracted


def test_archive_is_not_world_readable(tmp_path):
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)
    feed(oi, {}, "x" * 400)

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert not mode & stat.S_IRGRP
    assert not mode & stat.S_IROTH


def test_writes_stay_linear_in_the_size_of_the_output(tmp_path):
    """Each chunk must append only what is new.

    Rewriting the whole block per chunk is O(n^2): output arrives per line, so a
    few hundred KB over a few thousand lines turns into hundreds of MB written.
    """
    path = tmp_path / "outputs.log"
    oi = FakeInterpreter(path, max_output=100)

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
