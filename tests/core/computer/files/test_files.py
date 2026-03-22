import unittest
from unittest import mock

from interpreter.core.toolbox.files.files import Files, TextFileReader


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.files = Files(mock.Mock())

    @mock.patch("interpreter.core.toolbox.files.files.aifs")
    def test_search(self, mock_aifs):
        # Arrange
        mock_args = ["foo", "bar"]
        mock_kwargs = {"foo": "bar"}

        # Act
        self.files.search(mock_args, mock_kwargs)

        # Assert
        mock_aifs.search.assert_called_once_with(mock_args, mock_kwargs)

    def test_edit_original_text_in_filedata(self):
        # Arrange
        mock_open = mock.mock_open(read_data="foobar")
        mock_write = mock_open.return_value.write

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            self.files.edit("example/filepath/file", "foobar", "foobarbaz")

        # Assert
        mock_open.assert_any_call("example/filepath/file", "r")
        mock_open.assert_any_call("example/filepath/file", "w")
        mock_write.assert_called_once_with("foobarbaz")

    def test_edit_original_text_not_in_filedata(self):
        # Arrange
        mock_open = mock.mock_open(read_data="foobar")

        # Act
        with self.assertRaises(ValueError) as context_manager:
            with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
                self.files.edit("example/filepath/file", "barbaz", "foobarbaz")

        # Assert
        mock_open.assert_any_call("example/filepath/file", "r")
        self.assertEqual(
            str(context_manager.exception),
            "Original text not found. Did you mean one of these? foobar",
        )


class TestTextFileReader(unittest.TestCase):
    def setUp(self):
        self.files = Files(mock.Mock())
        self.test_content = "Line 1\nLine 2\nTODO: test\nLine 4\n### Section\nSection content"

    def test_get_reader(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")

        # Assert
        mock_open.assert_called_with("example.txt", "r", encoding="utf-8")
        self.assertIsInstance(reader, TextFileReader)

    @mock.patch("chardet.detect")
    def test_detect_encoding(self, mock_detect):
        mock_detect.return_value = {"encoding": "utf-8"}
        mock_open = mock.mock_open(read_data=self.test_content.encode("utf-8"))

        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt")

        mock_detect.assert_called_once()
        self.assertEqual(reader.encoding, "utf-8")

    def test_read_lines(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            lines = reader.read_lines(1, 3)
            lines_with_numbers = reader.read_lines(1, 3, show_line_numbers=True)

        # Assert
        self.assertEqual(lines, ["Line 1", "Line 2", "TODO: test"])
        self.assertEqual(lines_with_numbers, [(1, "Line 1"), (2, "Line 2"), (3, "TODO: test")])

    def test_read_characters(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            chars = reader.read_characters(0, 5)
            chars_with_numbers = reader.read_characters(0, 5, show_line_numbers=True)

        # Assert
        self.assertEqual(chars, "Line ")
        self.assertEqual(chars_with_numbers, [(0, 'L'), (1, 'i'), (2, 'n'), (3, 'e'), (4, ' ')])

    def test_search(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            matches = reader.search("TODO")
            matches_with_numbers = reader.search("TODO", show_line_numbers=True)

        # Assert
        self.assertEqual(matches, ["TODO: test"])
        self.assertEqual(matches_with_numbers, [(3, "TODO: test")])

    def test_filter_lines(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            filtered = reader.filter_lines(lambda x: "Line" in x)
            filtered_with_numbers = reader.filter_lines(lambda x: "Line" in x, show_line_numbers=True)

        # Assert
        self.assertEqual(filtered, ["Line 1", "Line 2", "Line 4"])
        self.assertEqual(filtered_with_numbers, [(1, "Line 1"), (2, "Line 2"), (4, "Line 4")])

    def test_find_section(self):
        # Arrange
        mock_open = mock.mock_open(read_data=self.test_content)

        # Act
        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            section = reader.find_section("### Section", lines_after=1)
            section_with_numbers = reader.find_section("### Section", lines_after=1, show_line_numbers=True)

        # Assert
        self.assertEqual(section, ["Section content"])
        self.assertEqual(section_with_numbers, [(6, "Section content")])

    @mock.patch("chardet.detect", return_value={"encoding": "utf-8", "confidence": 0.99})
    @mock.patch("interpreter.core.toolbox.files.files.os.path.getsize", return_value=100)
    def test_get_metadata(self, mock_getsize, mock_chardet_detect):
        mock_open = mock.mock_open(read_data=self.test_content.encode("utf-8"))

        with mock.patch("interpreter.core.toolbox.files.files.open", mock_open):
            reader = self.files.get_reader("example.txt", encoding="utf-8")
            metadata = reader.get_metadata()

        self.assertEqual(metadata["line_count"], 6)
