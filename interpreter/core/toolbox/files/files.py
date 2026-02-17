import difflib
import os
import re
from ...utils.lazy_import import lazy_import

# Lazy imports
aifs = lazy_import('aifs')

def _simple_detect_encoding(raw_data):
    """Simple encoding detection fallback when chardet isn't available.
    Checks for UTF-8 BOM and common UTF-8 patterns, defaults to 'utf-8'."""
    # Check for BOM
    if raw_data.startswith(b'\xef\xbb\xbf'):
        return {'encoding': 'utf-8', 'confidence': 1.0}
    # Check if it's valid UTF-8
    try:
        raw_data.decode('utf-8')
        return {'encoding': 'utf-8', 'confidence': 0.9}
    except UnicodeDecodeError:
        # If UTF-8 fails, try cp1252 (common on Windows)
        try:
            raw_data.decode('cp1252')
            return {'encoding': 'cp1252', 'confidence': 0.7}
        except UnicodeDecodeError:
            # Last resort
            return {'encoding': 'utf-8', 'confidence': 0.5}

class TextFileReader:
    def __init__(self, file_path, encoding='auto'):
        self.file_path = file_path
        self.encoding = encoding if encoding != 'auto' else self._detect_encoding()
        with open(file_path, 'r', encoding=self.encoding) as file:
            self.content = file.readlines()

    def _detect_encoding(self):
        """Auto-detect file encoding if not specified."""
        with open(self.file_path, 'rb') as file:
            raw_data = file.read()
        try:
            import chardet
            return chardet.detect(raw_data)['encoding']
        except ImportError:
            return _simple_detect_encoding(raw_data)['encoding']

    def _format_line(self, line_num, line, show_line_numbers=False):
        """Helper to format a line consistently for printing and returning"""
        line = line.strip()
        if show_line_numbers:
            print(f"{line_num}: {line}")
            return (line_num, line)
        print(line)
        return line

    def read_lines(self, from_line, to_line, show_line_numbers=False):
        """Read lines from `from_line` to `to_line` (1-based index).
        Prints lines immediately and returns them as a list."""
        lines = []
        for i, line in enumerate(self.content[from_line-1:to_line], start=from_line):
            lines.append(self._format_line(i, line, show_line_numbers))
        return lines

    def read_characters(self, from_char, to_char, show_line_numbers=False):
        """Read characters from `from_char` to `to_char` (0-based index).
        Prints characters immediately and returns them."""
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            content = file.read()
        content_chunk = content[from_char:to_char]

        if show_line_numbers:
            result = []
            for i, char in enumerate(content_chunk):
                print(f"{i}: {char}")
                result.append((i, char))
            return result

        print(content_chunk)
        return content_chunk

    def search(self, pattern, show_line_numbers=False):
        """Search for lines matching a regex pattern.
        Prints matches immediately and returns them as a list.

        Example:
            reader.search(r"TODO:.*")  # Find all TODO items
        """
        matches = []
        for i, line in enumerate(self.content, start=1):
            if re.search(pattern, line):
                matches.append(self._format_line(i, line, show_line_numbers))
        return matches

    def filter_lines(self, condition, show_line_numbers=False):
        """Filter lines using a custom Python function/lambda.
        Prints matching lines immediately and returns them as a list.

        Example:
            reader.filter_lines(lambda line: "TODO" in line and "urgent" in line.lower())
        """
        filtered = []
        for i, line in enumerate(self.content, start=1):
            if condition(line):
                filtered.append(self._format_line(i, line, show_line_numbers))
        return filtered

    def find_section(self, section_name, lines_after=10, show_line_numbers=False):
        """Find a line containing the given text and return subsequent lines.
        Useful for finding sections in any text file (markdown headers, code comments, etc.).
        Prints matching section immediately and returns lines as a list.

        Example:
            reader.find_section("### Installation", lines_after=5)  # Find markdown section
            reader.find_section("# Configuration", lines_after=20)  # Find code comment section
        """
        result = []
        for i, line in enumerate(self.content):
            if section_name in line:
                start = i + 1
                end = min(start + lines_after, len(self.content))
                for j, section_line in enumerate(self.content[start:end], start=start):
                    result.append(self._format_line(j, section_line, show_line_numbers))
                break
        return result

    def get_metadata(self):
        """Get basic metadata about the file."""
        # Get file size in bytes
        file_size = os.path.getsize(self.file_path)

        # Count total characters (including whitespace)
        total_chars = sum(len(line) for line in self.content)

        # Count non-whitespace characters
        non_whitespace_chars = sum(len(line.strip()) for line in self.content)

        # Try to get encoding confidence
        with open(self.file_path, 'rb') as file:
            raw_data = file.read()
        try:
            import chardet
            confidence = chardet.detect(raw_data)['confidence']
        except ImportError:
            confidence = _simple_detect_encoding(raw_data)['confidence']

        metadata = {
            'path': self.file_path,
            'encoding': self.encoding,
            'line_count': len(self.content),
            'file_size_bytes': file_size,
            'total_chars': total_chars,
            'non_whitespace_chars': non_whitespace_chars,
            'confidence': confidence
        }

        print(f"File: {metadata['path']}")
        print(f"Encoding: {metadata['encoding']} (confidence: {metadata['confidence']:.2%})")
        print(f"Lines: {metadata['line_count']}")
        print(f"Size: {metadata['file_size_bytes']:,} bytes")
        print(f"Characters: {metadata['total_chars']:,} (non-whitespace: {metadata['non_whitespace_chars']:,})")

        return metadata


class Files:
    def __init__(self, toolbox):
        self.toolbox = toolbox

    def get_reader(self, path, encoding='auto'):
        """
        Get a TextFileReader instance for the specified file path.
        Provides convenient methods for reading and analyzing text files.

        Args:
            path (str): Path to the text file
            encoding (str, optional): File encoding. Use 'auto' for automatic detection (default).

        Returns:
            TextFileReader: A reader instance for the specified file

        Example:
            ```python
            reader = toolbox.files.get_reader("example.txt")
            first_10_lines = reader.read_lines(1, 10)
            matches = reader.search("TODO:")
            ```
        """
        # Print helpful usage message
        print(f"\nCreated reader for '{path}'. Methods: read_lines(from, to), read_characters(from, to), search(pattern), filter_lines(condition), find_section(name), get_metadata() - all support show_line_numbers=False and print+return results")
        return TextFileReader(path, encoding)

    def search(self, *args, **kwargs):
        """
        Semantic search over file contents: pass query and optionally path; requires aifs library.

        Forwards to the 'aifs' package: chunks and embeds file contents, then returns
        the text chunks that best match the query (by embedding similarity). First run
        indexes the path and can be slow; later runs reuse the index.

        Requires the 'aifs' package (`pip install aifs`). If aifs is not installed,
        this method raises ImportError with install instructions. Install implications:
        aifs uses chroma and a local embedding model (download on first use). For
        parsing PDF, Office, images, etc., the aifs README recommends
        `pip install "unstructured[all-docs]"` (includes large packages). All
        arguments are passed through to aifs.search(); see the aifs package for the
        current API (e.g. path, file_paths, max_results, verbose, python_docstrings_only).

        Args:
            query (str): Natural-language or keyword search query.
            *args, **kwargs: Passed through to aifs.search().

        Returns:
            list: Matching text chunks (strings), ordered by relevance (per aifs).

        Example:
            >>> toolbox.files.search("where is the login logic", path="src")
        """
        if aifs is None:
            raise ImportError(
                "Semantic file search requires the 'aifs' package. Install with: pip install aifs\n"
                "Note: aifs uses chroma and downloads an embedding model on first use. For PDF/Office/images, "
                "see https://github.com/openinterpreter/aifs — optional pip install \"unstructured[all-docs]\" (large)."
            )
        return aifs.search(*args, **kwargs)

    def edit(self, path, original_text, replacement_text):
        """
        Edits a file on the filesystem, replacing the original text with the replacement text.

        Returns:
            None
        """
        with open(path, "r") as file:
            filedata = file.read()

        if original_text not in filedata:
            matches = get_close_matches_in_text(original_text, filedata)
            if matches:
                suggestions = ", ".join(matches)
                raise ValueError(
                    f"Original text not found. Did you mean one of these? {suggestions}"
                )

        filedata = filedata.replace(original_text, replacement_text)

        with open(path, "w") as file:
            file.write(filedata)


def get_close_matches_in_text(original_text, filedata, n=3):
    """
    Returns the closest matches to the original text in the content of the file.
    """
    words = filedata.split()
    original_words = original_text.split()
    len_original = len(original_words)

    matches = []
    for i in range(len(words) - len_original + 1):
        phrase = " ".join(words[i : i + len_original])
        similarity = difflib.SequenceMatcher(None, original_text, phrase).ratio()
        matches.append((similarity, phrase))

    matches.sort(reverse=True)
    return [match[1] for match in matches[:n]]
