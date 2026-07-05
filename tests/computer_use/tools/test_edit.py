import importlib.util
import sys
from pathlib import Path

import pytest

_tools = Path(__file__).resolve().parents[3] / "interpreter/computer_use/tools"

for mod_name, filename in [
    ("interpreter.computer_use.tools.base", "base.py"),
    ("interpreter.computer_use.tools.run", "run.py"),
    ("interpreter.computer_use.tools.edit", "edit.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, _tools / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

ToolError = sys.modules["interpreter.computer_use.tools.base"].ToolError
EditTool = sys.modules["interpreter.computer_use.tools.edit"].EditTool


def test_validate_path_requires_absolute():
    """EditTool.validate_path rejects relative paths because file operations require an absolute location."""
    tool = EditTool()
    with pytest.raises(ToolError, match="not an absolute path"):
        tool.validate_path("view", Path("relative.txt"))


def test_validate_path_create_requires_missing_file(tmp_path):
    """The create command accepts only paths that do not yet exist and rejects paths to existing files."""
    tool = EditTool()
    new_file = tmp_path / "new.txt"
    tool.validate_path("create", new_file)
    new_file.write_text("exists")
    with pytest.raises(ToolError, match="already exists"):
        tool.validate_path("create", new_file)


def test_str_replace_replaces_unique_match(tmp_path):
    """str_replace updates the file when the old string appears exactly once and reports success."""
    tool = EditTool()
    path = tmp_path / "file.txt"
    path.write_text("hello world")
    result = tool.str_replace(path, "world", "there")
    assert "hello there" in path.read_text()
    assert "edited" in result.output


def test_str_replace_rejects_missing_string(tmp_path):
    """str_replace raises ToolError when the old string is not found verbatim in the file."""
    tool = EditTool()
    path = tmp_path / "file.txt"
    path.write_text("hello")
    with pytest.raises(ToolError, match="did not appear verbatim"):
        tool.str_replace(path, "missing", "x")


def test_str_replace_rejects_multiple_matches(tmp_path):
    """str_replace raises ToolError on multiple matches to prevent ambiguous or unintended bulk replacements."""
    tool = EditTool()
    path = tmp_path / "file.txt"
    path.write_text("aa\naa")
    with pytest.raises(ToolError, match="Multiple occurrences"):
        tool.str_replace(path, "aa", "b")


def test_insert_adds_lines_at_position(tmp_path):
    """insert inserts new text at the given zero-based line index without disturbing surrounding lines."""
    tool = EditTool()
    path = tmp_path / "file.txt"
    path.write_text("line1\nline3")
    result = tool.insert(path, 1, "line2")
    assert path.read_text().splitlines() == ["line1", "line2", "line3"]
    assert "edited" in result.output


def test_undo_edit_restores_previous_content(tmp_path):
    """undo_edit reverts the file to its content before the most recent edit using the edit history."""
    tool = EditTool()
    path = tmp_path / "file.txt"
    path.write_text("original")
    tool.str_replace(path, "original", "changed")
    result = tool.undo_edit(path)
    assert path.read_text() == "original"
    assert "undone successfully" in result.output


def test_make_output_includes_line_numbers():
    """_make_output formats file content with tab-separated line numbers so the model can reference specific lines."""
    tool = EditTool()
    output = tool._make_output("alpha\nbeta", "/tmp/demo.txt", init_line=1)
    assert "alpha" in output
    assert "beta" in output
    assert "1\t" in output
