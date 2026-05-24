"""Map target file paths to Pygments lexer names (file format, not edit language)."""

from pathlib import Path

TARGET_FILE_EXT_LEXER = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".sh": "bash",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sql": "sql",
    ".toml": "toml",
    ".r": "r",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
}


def syntax_lang_for_target_path(target):
    return TARGET_FILE_EXT_LEXER.get(Path(target).suffix.lower(), "text")


def syntax_lang_for_dry_run(target, edit_language):
    if edit_language == "patch":
        return "diff"
    return syntax_lang_for_target_path(target)
