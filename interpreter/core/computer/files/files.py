import difflib

from ...utils.lazy_import import lazy_import

# Lazy import of aifs, imported when needed
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

    def read_lines(self, from_line, to_line, show_line_numbers=False):
        """Read lines from `from_line` to `to_line` (1-based index).
        Optionally show line numbers."""
        lines = self.content[from_line-1:to_line]
        if show_line_numbers:
            return [(i + from_line, line.strip()) for i, line in enumerate(lines)]
        return [line.strip() for line in lines]

    def read_characters(self, from_char, to_char, show_line_numbers=False):
        """Read characters from `from_char` to `to_char` (0-based index).
        Optionally show line numbers."""
        with open(self.file_path, 'r', encoding=self.encoding) as file:
            content = file.read()
        content_chunk = content[from_char:to_char]
        if show_line_numbers:
            return [(i, char) for i, char in enumerate(content_chunk)]
        return content_chunk

    def search(self, pattern, show_line_numbers=False):
        """Search for pattern in the file and return matching lines."""
        matches = []
        for i, line in enumerate(self.content):
            if re.search(pattern, line):
                if show_line_numbers:
                    matches.append((i + 1, line.strip()))  # Return 1-based line number
                else:
                    matches.append(line.strip())
        return matches

    def filter_lines(self, condition, show_line_numbers=False):
        """Filter lines based on a condition."""
        filtered = [line for line in self.content if condition(line)]
        if show_line_numbers:
            return [(i + 1, line.strip()) for i, line in enumerate(filtered)]
        return [line.strip() for line in filtered]

    def find_section(self, section_name, lines_after=10, show_line_numbers=False):
        """Find a section by name (e.g., '### To do') and return subsequent lines."""
        for i, line in enumerate(self.content):
            if section_name in line:
                start = i + 1
                return self.read_lines(start, start + lines_after, show_line_numbers)
        return []  # Section not found

    def get_metadata(self):
        """Get basic metadata about the file."""
        return {
            'line_count': len(self.content),
            'file_size': len(self.content),  # This can be changed to actual file size in bytes
        }

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
