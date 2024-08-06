from typing import Callable

import regex

from comment_parser.comment_parser import (
    extract_comments_from_str as cp_extract_comments_from_str,
    UnsupportedError as CPUnsupportedError,
)


def remove_non_ASCII_characters(text: str):
    """
    This function removes non-ASCII characters from a string.

    The resulting string will only contain characters in the ASCII character set, including spaces and punctuation marks.

    Args:
        text (str): The input string that needs to be stripped of non-ASCII characters.

    Returns:
        str: A new string containing only ASCII characters.
    """
    result = regex.sub(r"[^\x00-\x7F]+", "", text)
    # without control characters:
    # result = regex.sub(r'[^\x20-\x7E]+', '', text)
    return result


def remove_non_ISO_8859_1_characters(text: str):
    """
    This function removes non-ISO 8859-1 characters from a string.

    The resulting string will only contain characters in the ISO 8859-1
    character set, including spaces and punctuation marks.

    Args:
        text (str): The input string that needs to be stripped of non-ISO 8859-1 characters.

    Returns:
        str: A new string containing only ISO 8859-1 characters.
    """
    # result = regex.sub(r'[^\p{Latin}]', '', text)
    result = regex.sub(r"[^\x00-\x7F\xA0-\xFF]+", "", text)
    return result


def remove_non_ISO_8859_1_characters_in_comments(text: str, language: str) -> str:
    """
    This function removes non-ISO 8859-1 characters from comments in a given text string.
    It checks the language of the input text string before removing any characters to
    avoid accidentally removing valid code.

    Args:
        text (str): The input text string containing comments.
        language (str): The language of the input text string.

    Returns:
        str: The cleaned text string with non-ISO 8859-1 characters removed from comments.
    """
    comments = cp_extract_comments_from_str(text, map_language_to_mime(language))
    result = ""
    start = 0
    end = 0
    for c in comments:
        # comment_parser.Comment.line_number() returns 1-indexed line numbers not 0-indexed
        end = c.line_number() - 1
        # code
        result += text_between_lines(text, start, end)
        # comments / code with inline comment
        remaining_text = text_between_lines(text, end, None)
        cleaned_text, offset = replace_lines_in_string_with_ISO_8859_1_conform_substrings(remaining_text, c.text())
        result += cleaned_text
        # go to next line
        start = end + offset + 1
    result += text_between_lines(text, start, None)
    return result


def remove_non_ASCII_characters_in_comments(text: str, language: str) -> str:
    """
    This function removes non-ASCII characters from comments in a given text string.
    It checks the language of the input text string before removing any characters to
    avoid accidentally removing valid code.

    Args:
        text (str): The input text string containing comments.
        language (str): The language of the input text string.

    Returns:
        str: The cleaned text string with non-ASCII characters removed from comments.
    """
    comments = cp_extract_comments_from_str(text, map_language_to_mime(language))
    result = ""
    start = 0
    end = 0
    for c in comments:
        # comment_parser.Comment.line_number() returns 1-indexed line numbers not 0-indexed
        end = c.line_number() - 1
        # code
        result += text_between_lines(text, start, end)
        # comments / code with inline comment
        remaining_text = text_between_lines(text, end, None)
        cleaned_text, offset = replace_lines_in_string_with_ASCII_conform_substrings(remaining_text, c.text())
        result += cleaned_text
        # print("----------------------------", end, end+offset)
        # print(result)
        # go to next line
        start = end + offset + 1
    result += text_between_lines(text, start, None)
    return result


def remove_comments_from_code(text: str, language: str) -> str:
    """Remove all text inside comments from the code in the given programming language."""
    comments = cp_extract_comments_from_str(text, map_language_to_mime(language))
    result = ""
    start = 0
    end = 0
    for c in comments:
        # comment_parser.Comment.line_number() returns 1-indexed line numbers not 0-indexed
        end = c.line_number() - 1
        # code
        result += text_between_lines(text, start, end)
        # comments / code with inline comment
        remaining_text = text_between_lines(text, end, None)
        cleaned_text, offset = delete_substring(remaining_text, c.text())
        result += cleaned_text
        # go to next line
        start = end + offset + 1
    result += text_between_lines(text, start, None)
    return result


def text_between_lines(text: str, start: int, stop: int | None) -> str:
    """Text from given `start` line to the `stop`-1 line not including index `stop`.
    If `stop` is None: the text is returned from start up to the end."""
    lines = text.splitlines(keepends=True)
    if stop is None:
        return "".join(lines[start:])
    return "".join(lines[start:stop])


def string_replace_lines(original_string: str, string_to_replace: str, deletion_pattern: str) -> tuple[str, int]:
    """
    Find a `string_to_replace` in a string and replace it with a version of it where a given pattern is deleted from.
    The `string_to_replace` should start in the first line of the original string.
    Returns the resulting string and the last line in the string that contained the string to replace.
    """
    list_old = original_string.splitlines(keepends=True)
    list_to_modify = string_to_replace.splitlines(keepends=True)

    result = ""
    for index, (original_line, substring_to_modify) in enumerate(zip(list_old, list_to_modify)):
        modified_substring = regex.sub(deletion_pattern, "", substring_to_modify)
        result += original_line.replace(substring_to_modify, modified_substring)
    return result, index


def delete_substring(original_string: str, string_to_delete: str) -> tuple[str, int]:
    """
    Find a `string_to_delete` in a string and delete it.
    The `string_to_delete` should start in the first line of the original string.
    Returns the resulting string and the last line in the string that contained the string to replace.
    """
    list_old = original_string.splitlines(keepends=True)
    list_to_modify = string_to_delete.splitlines(keepends=True)

    result = ""
    for index, (original_line, substring_to_modify) in enumerate(zip(list_old, list_to_modify)):
        result += original_line.replace(substring_to_modify, "")
    return result, index


def replace_lines_in_string_with_charset_conform_substrings(
    original_string: str, string_to_replace: str, remove_non_charset_characters: Callable
) -> tuple[str, int]:
    """
    Find a `string_to_replace` in a string and replace it with a version of it that is cleaned to only characters in a given charset.
    The removal of non-charset conform characters is defined by the given function `remove_non_charset_characters`.
    The `string_to_replace` should start in the first line of the original string.
    Returns the resulting string and the last line in the string that contained the string to replace.
    """
    list_old = original_string.splitlines(keepends=True)
    list_to_modify = string_to_replace.splitlines(keepends=True)

    result = ""
    index = 0
    for index, (original_line, substring_to_modify) in enumerate(zip(list_old, list_to_modify)):
        result += original_line.replace(substring_to_modify, remove_non_charset_characters(substring_to_modify))
    return result, index


def replace_lines_in_string_with_ISO_8859_1_conform_substrings(
    original_string: str, string_to_replace: str
) -> tuple[str, int]:
    """
    Find a `string_to_replace` in a string and replace it with a version of it that is cleaned to only Latin-1 characters.
    The `string_to_replace` should start in the first line of the original string.
    Returns the resulting string and the last line in the string that contained the string to replace.
    """
    # list_old = original_string.splitlines(keepends=True)
    # list_to_modify = string_to_replace.splitlines(keepends=True)

    # result = ""
    # for index, (original_line, substring_to_modify) in enumerate(zip(list_old, list_to_modify)):
    #     result += original_line.replace(substring_to_modify, remove_non_ISO_8859_1_characters(substring_to_modify))
    # return result, index
    return replace_lines_in_string_with_charset_conform_substrings(
        original_string, string_to_replace, remove_non_ISO_8859_1_characters
    )


def replace_lines_in_string_with_ASCII_conform_substrings(
    original_string: str, string_to_replace: str
) -> tuple[str, int]:
    """
    Find a `string_to_replace` in a string and replace it with a version of it that is cleaned to only ASCII characters.
    The `string_to_replace` should start in the first line of the original string.
    Returns the resulting string and the last line in the string that contained the string to replace.
    """
    return replace_lines_in_string_with_charset_conform_substrings(
        original_string, string_to_replace, remove_non_ASCII_characters
    )


def map_language_to_mime(language: str) -> str:
    """Map programming language to MIME type as used by comment_parser."""
    MIME_MAP = {
        "Javascript": "application/javascript",
        "HTML": "text/html",
        "C": "text/x-c",
        "C++": "text/x-c++",
        "C#": "text/x-c++",
        "Go": "text/x-go",
        "Java": "text/x-java",
        "Rust": "text/x-java",
        "Java": "text/x-java-source",
        "Javascript": "text/x-javascript",
        "Python": "text/x-python",
        "Ruby": "text/x-ruby",
        "Python": "text/x-script.python",
        "Shell": "text/x-shellscript",
        "XML": "text/xml",
    }

    if language not in MIME_MAP.keys():
        raise CPUnsupportedError("comment_parser does not support the requested language:", language)
    return MIME_MAP[language]
