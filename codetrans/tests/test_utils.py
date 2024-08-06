import unittest

from codetrans import remove_non_ISO_8859_1_characters_in_comments
from codetrans.code_encoding_sanitization import (
    remove_non_ISO_8859_1_characters,
    remove_comments_from_code,
    text_between_lines,
    string_replace_lines,
    delete_substring,
    replace_lines_in_string_with_ISO_8859_1_conform_substrings,
)


class TestRemoveNonISO88591Characters(unittest.TestCase):

    def test_remove_special_characters(self):
        # Input string containing non-ISO 8859-1 characters
        text = """//PN:全ケースにおける、部分木Sに含まれる頂点数の総和"""

        # Expected output string containing only ISO 8859-1 characters
        expected_output = "//PN:S"

        # Actual output string returned by the function
        actual_output = remove_non_ISO_8859_1_characters(text)

        # Compare expected and actual output strings
        self.assertEqual(actual_output, expected_output)


def test_remove_non_iso_8859_1_characters_in_comments():
    text_1 = """# PN:全ケースにおける、部分木Sに含まれる頂点数の総和\nprint("Hello")"""
    language = "Python"
    cleaned_text = remove_non_ISO_8859_1_characters_in_comments(text_1, language)
    assert cleaned_text == """# PN:S\nprint("Hello")"""

    text_2 = """print("Hello") # PN:全ケースにおける、部分木Sに含まれる頂点数の総和"""
    language = "Python"
    cleaned_text = remove_non_ISO_8859_1_characters_in_comments(text_2, language)
    assert cleaned_text == """print("Hello") # PN:S"""

    text_3 = """print("Hello") # PN:全ケースにおける、部分木Sに含まれる頂点数の総和\nprint(1) # hallo点"""
    language = "Python"
    cleaned_text = remove_non_ISO_8859_1_characters_in_comments(text_3, language)
    assert cleaned_text == """print("Hello") # PN:S\nprint(1) # hallo"""

    text_4 = """\n# This is a comment点点点\nprint(1) # Another 点点点comment"""
    language = "Python"
    cleaned_text = remove_non_ISO_8859_1_characters_in_comments(text_4, language)
    assert cleaned_text == """\n# This is a comment\nprint(1) # Another comment"""

    text_5 = '\n"""multiline\n docstring 点点点 comment"""\nprint(1)\n'
    cleaned_text = remove_non_ISO_8859_1_characters_in_comments(text_5, "Python")
    assert cleaned_text == '\n"""multiline\n docstring  comment"""\nprint(1)\n'


def test_delete_substring():
    example_string = "This is a sample:\nIs\na sample\nString"
    result, index = delete_substring(example_string, "sample")
    assert result == "This is a :\n"
    assert index == 0


def test_remove_comments_from_code():
    """Tests the `remove_comments_from_code` function."""
    text_1 = """# PN:全ケースにおける、部分木Sに含まれる頂点数の総和\nprint("Hello")"""
    language_1 = "Python"
    actual = remove_comments_from_code(text_1, language_1)
    assert actual == """#\nprint("Hello")"""

    text_2 = """# print("Hello")\nprint("Hello")"""
    language_2 = "Python"
    actual = remove_comments_from_code(text_2, language_2)
    assert actual == """#\nprint("Hello")"""

    text_3 = """print("Hello") # PN:全ケースにおける、部分木Sに含まれる頂点数の総和\nprint(1) # hallo点"""
    language = "Python"
    cleaned_text = remove_comments_from_code(text_3, language)
    assert cleaned_text == """print("Hello") #\nprint(1) #"""

    # Test removing comments from a simple Python script with single-line comments
    text_4 = """
# This is a comment
print(1) # Another comment
"""
    expected_output_4 = """
#
print(1) #
"""
    assert remove_comments_from_code(text_4, "Python") == expected_output_4


def test_remove_comments_from_code_with_inline_comment():
    # Test removing comments from a Python script with inline comment in a block of code
    input_text = """
# This is a comment
x = 1 # Another comment
y = x * 2
print(y) # Yet another comment
"""
    expected_output = """
#
x = 1 #
y = x * 2
print(y) #
"""
    assert remove_comments_from_code(input_text, "Python") == expected_output


# TODO check if comment_parser can detect docstring format for Python
def test_remove_comments_from_code_with_multiline_comment():
    # Test removing multiline comments from a Python script
    input_text = '\n"""multiline\n docstring comment"""\nprint(1)\n'
    expected_output = '\n""""""\nprint(1)\n'
    assert remove_comments_from_code(input_text, "Python") == expected_output

    input_text = "\n/*multiline\ncomment*/\nint a = 0;\n"
    expected_output = "\n/**/\nint a = 0;\n"
    assert remove_comments_from_code(input_text, "Java") == expected_output


def test_remove_non_ISO88591_characters():
    # Test case 1: removing non-ISO 8859-1 characters from a string containing only valid ISO 8859-1 characters
    result = remove_non_ISO_8859_1_characters("Hello, world!")
    assert result == "Hello, world!"

    # Test case 2: removing non-ISO 8859-1 characters from a string containing invalid ISO 8859-1 characters
    result = remove_non_ISO_8859_1_characters("部分木")
    assert result == ""

    # Test case 3: removing non-ISO 8859-1 characters from a string containing mixed ISO 8859-1 and invalid characters
    result = remove_non_ISO_8859_1_characters("Hello, World? 部分木")
    assert result == "Hello, World? "


def test_text_between_lines():
    assert (
        text_between_lines("This is line 1\nThis is line 2\nThis is line 3", 1, 3) == "This is line 2\nThis is line 3"
    )
    assert (
        text_between_lines("This is line 1\nThis is line 2\nThis is line 3\nThis is line 4", 1, 3)
        == "This is line 2\nThis is line 3\n"
    )
    assert text_between_lines("This is line 1\nThis is line 2\nThis is line 3\n", 3, 3) == ""
    assert (
        text_between_lines("This is line 1\nThis is line 2\nThis is line 3\n", 0, 3)
        == "This is line 1\nThis is line 2\nThis is line 3\n"
    )
    assert (
        text_between_lines("This is line 1\nThis is line 2\nThis is line 3\n", 1, None)
        == "This is line 2\nThis is line 3\n"
    )


def test_string_replace_lines():
    assert string_replace_lines("foo\nbar\nbaz\n", "foo\nbar\nbaz\n", "a") == ("foo\nbr\nbz\n", 2)
    assert string_replace_lines("foo\nbar\nbaz\n", "foo\nbar\n", "a") == ("foo\nbr\n", 1)
    assert string_replace_lines("foo\nbar\nbaz\n", "foo", "f") == ("oo\n", 0)
    assert string_replace_lines("foo bar\nbaz\n", "bar\nbaz\n", "b") == ("foo ar\naz\n", 1)


def test_replace_lines_in_string_with_ISO_8859_1_conform_substrings():
    original_string = "print(1) # PN:全ケースにおける、部分木\nSに含まれる頂点数の総和"
    string_to_replace = "# PN:全ケースにおける、部分木\nSに含まれる頂点数の総和"
    result, index = replace_lines_in_string_with_ISO_8859_1_conform_substrings(original_string, string_to_replace)
    assert result == "print(1) # PN:\nS"
    assert index == 1
