import difflib
import os
import re
from ...utils.lazy_import import lazy_import

# Lazy imports
aifs = lazy_import('aifs')
chardet = lazy_import('chardet')


class TextFileReader:
    def __init__(self, file_path, encoding=None):
        self.file_path = file_path
        self.encoding = encoding or self._detect_encoding()
        with open(file_path, 'r', encoding=self.encoding) as file:
            self.content = file.readlines()

    def _detect_encoding(self):
        """Auto-detect file encoding if not specified."""
        with open(self.file_path, 'rb') as file:
            raw_data = file.read()
        return chardet.detect(raw_data)['encoding']

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
        """Search for pattern in the file and return matching lines.
        Prints matches immediately and returns them as a list."""
        matches = []
        for i, line in enumerate(self.content, start=1):
            if re.search(pattern, line):
                matches.append(self._format_line(i, line, show_line_numbers))
        return matches

    def filter_lines(self, condition, show_line_numbers=False):
        """Filter lines based on a condition.
        Prints matching lines immediately and returns them as a list."""
        filtered = []
        for i, line in enumerate(self.content, start=1):
            if condition(line):
                filtered.append(self._format_line(i, line, show_line_numbers))
        return filtered

    def find_section(self, section_name, lines_after=10, show_line_numbers=False):
        """Find a section by name (e.g., '### To do') and return subsequent lines.
        Prints matching section immediately and returns lines as a list."""
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

        metadata = {
            'path': self.file_path, 'encoding': self.encoding,
            'line_count': len(self.content),
            'file_size_bytes': file_size, 'total_chars': total_chars,
            'non_whitespace_chars': non_whitespace_chars,
            'confidence': chardet.detect(open(self.file_path, 'rb').read())
            ['confidence']}

        print(f"File: {metadata['path']}")
        print(f"Encoding: {metadata['encoding']} (confidence: {metadata['confidence']:.2%})")
        print(f"Lines: {metadata['line_count']}")
        print(f"Size: {metadata['file_size_bytes']:,} bytes")
        print(f"Characters: {metadata['total_chars']:,} (non-whitespace: {metadata['non_whitespace_chars']:,})")

        return metadata


class Files:
    def __init__(self, computer):
        self.computer = computer

    def get_reader(self, path, encoding=None):
        """
        Get a TextFileReader instance for the specified file path.
        Provides convenient methods for reading and analyzing text files.

        Args:
            path (str): Path to the text file
            encoding (str, optional): File encoding. Will auto-detect if not specified.

        Returns:
            TextFileReader: A reader instance for the specified file

        Example:
            ```python
            reader = computer.files.get_reader("example.txt")
            first_10_lines = reader.read_lines(1, 10)
            matches = reader.search("TODO:")
            ```
        """
        return TextFileReader(path, encoding)

    def search(self, *args, **kwargs):
        """
        Search the filesystem for the given query.
        """
        return aifs.search(*args, **kwargs)

    def edit(self, path, original_text, replacement_text):
        """
        Edits a file on the filesystem, replacing the original text with the replacement text.
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
