from bidict import bidict
import pytest
import unittest

from codetrans.postprocessing import (
    Balance,
    extract_markdown_code_blocks_with_pl,
    pl_from_markdown_pl_tag,
    markdown_pl_mapping_bidict,
    semicolon_syntax,
    comment_syntax,
    extract_markdown_code_blocks,
    extract_markdown_code_blocks_code_only,
    check_balance_of_code_fences,
)


def test_semicolon_syntax_valid_languages():
    assert semicolon_syntax("Java") is True
    assert semicolon_syntax("Go") is True
    assert semicolon_syntax("Rust") is True
    assert semicolon_syntax("C") is True
    assert semicolon_syntax("C++") is True
    assert semicolon_syntax("C#") is True
    assert semicolon_syntax("Python") is False


def test_semicolon_syntax_invalid_language():
    with pytest.raises(NotImplementedError):
        semicolon_syntax("PHP")
    with pytest.raises(NotImplementedError):
        semicolon_syntax("Swift")


def test_comment_syntax():
    # Testing correct syntax
    assert comment_syntax("Python") == ["#", '"""']
    assert comment_syntax("Java") == ["//", "/*"]
    assert comment_syntax("Go") == ["//", "/*"]
    assert comment_syntax("Rust") == ["//", "/*"]
    assert comment_syntax("C") == ["//", "/*"]
    assert comment_syntax("C++") == ["//", "/*"]
    assert comment_syntax("C#") == ["//", "/*"]

    # Testing missing language
    with pytest.raises(NotImplementedError):
        comment_syntax("Lua")


def test_extract_markdown_code_blocks_empty_string():
    assert extract_markdown_code_blocks("") == []


def test_extract_markdown_code_blocks_no_code_blocks():
    assert extract_markdown_code_blocks("This is a string with no code blocks.") == []


def test_extract_markdown_code_blocks_incomplete_code_block():
    input_text = "```python\ndef add(x, y):\n    return x + y\n"
    assert extract_markdown_code_blocks(input_text) == []


def test_extract_markdown_code_blocks_single_code_block():
    input_text = "```python\ndef add(x, y):\n    return x + y\n```"
    assert extract_markdown_code_blocks(input_text) == [
        ("python", "def add(x, y):\n    return x + y")
    ]


def test_extract_markdown_code_blocks_multiple_code_blocks():
    input_text = "This is a string with multiple code blocks.\n\n```python\ndef add(x, y):\n    return x + y\n```\nAfter the code block."
    assert extract_markdown_code_blocks(input_text) == [
        ("python", "def add(x, y):\n    return x + y")
    ]


def test_extract_markdown_code_blocks_mixed_syntax_code_blocks():
    input_text = "This is a string with mixed syntax code blocks.\n\n```python\ndef add(x, y):\n    return x + y\n```\nAnother code block.\n\n```yaml\nkey1: value1\n```\nAfter the code block."
    assert extract_markdown_code_blocks(input_text) == [
        ("python", "def add(x, y):\n    return x + y"),
        ("yaml", "key1: value1"),
    ]


def test_extract_markdown_code_blocks_in_language():
    text = """
# Python code block
Here's some Python code:
```python
print("Hello, world!")
```
# Java code block
Here's some Java code:
```java
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
}
```
"""
    assert extract_markdown_code_blocks_with_pl(text, "Python") == [
        'print("Hello, world!")'
    ]
    assert extract_markdown_code_blocks_with_pl(text, "C") == []

    with pytest.raises(NotImplementedError):
        extract_markdown_code_blocks_with_pl(text, "notexisting")


def test_extract_markdown_code_blocks_code_only():
    # empty string
    assert extract_markdown_code_blocks_code_only("") == []

    # no code blocks
    assert (
        extract_markdown_code_blocks_code_only("This is a string with no code blocks.")
        == []
    )

    # single code block
    input_text = "```python\ndef add(x, y):\n    return x + y\n```"
    assert extract_markdown_code_blocks_code_only(input_text) == [
        "def add(x, y):\n    return x + y"
    ]

    # single code block surrounded by text
    input_text = "This is a string with multiple code blocks.\n\n```python\ndef add(x, y):\n    return x + y\n```\nAfter the code block."
    assert extract_markdown_code_blocks_code_only(input_text) == [
        "def add(x, y):\n    return x + y"
    ]

    # multiple code blocks
    input_text = "This is a string with multiple code blocks.\n\n```python\ndef add(x, y):\n    return x + y\n```\Between the code block.\n```python\ndef sub(x, y):\n    return x - y\n```"
    assert extract_markdown_code_blocks_code_only(input_text) == [
        "def add(x, y):\n    return x + y",
        "def sub(x, y):\n    return x - y",
    ]

    # mixed syntax codeblocks
    input_text = "This is a string with mixed syntax code blocks.\n\n```python\ndef add(x, y):\n    return x + y\n```\nAnother code block.\n\n```yaml\nkey1: value1\n```\nAfter the code block."
    assert extract_markdown_code_blocks_code_only(input_text) == [
        "def add(x, y):\n    return x + y",
        "key1: value1",
    ]


def test_markdown_language_mapping():
    assert type(markdown_pl_mapping_bidict()) == bidict
    assert markdown_pl_mapping_bidict()["Python"] == "python"
    assert markdown_pl_mapping_bidict()["C++"] == "cpp"


def test_get_language_from_markdown_language_identifier():
    assert pl_from_markdown_pl_tag("python") == "Python"
    assert pl_from_markdown_pl_tag("java") == "Java"
    assert pl_from_markdown_pl_tag("cpp") == "C++"
    assert pl_from_markdown_pl_tag("csharp") == "C#"
    assert pl_from_markdown_pl_tag("rust") == "Rust"

    # Testing missing language
    with pytest.raises(NotImplementedError):
        pl_from_markdown_pl_tag("thisdoesnotexist")

    with pytest.raises(NotImplementedError):
        pl_from_markdown_pl_tag("lua")


def test_check_balance_of_code_fences():
    # empty string
    text = ""
    language = None
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.NO_MARKDOWN}

    # no markdown
    text = """print("1")"""
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.NO_MARKDOWN}

    # Test for an empty string with no language identifier
    text = "```"
    language = None
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.CLOSING_MISSING}

    # Balanced with language tags
    text = """
```python
print("Hello World")
```

```python
print("Hi")
```
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.BALANCED}

    # Balanced with Text
    text = """
Here is code:
```python
print("Hello World")
```
Some more code:
```python
print("Hi")
```
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.BALANCED}

    # Balanced with Text and ``` as opening tag
    text = """
Here is code:
```
print("Hello World")
```
Some more code:
```python
print("Hi")
```
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.BALANCED}

    # Closing missing
    text = """
```python
print("Hello World")
```
```java
package org.example;

    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.CLOSING_MISSING}

    # Closing missing
    text = """
Sure here is the code:
```python
print("Hello World")
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.CLOSING_MISSING}

    text = """
```python
print("Hello World")

```java
String a = "Hello";
```
```
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.NESTED}

    # Opening missing
    text = """

print("Hello World")
```  
Output:
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.OPENING_MISSING}

    # Opening missing
    text = """
```
This is some text.
```
print("Hello World")
```
Output:
    """
    language = "Python"
    result = check_balance_of_code_fences(text, language)
    assert result == {Balance.OPENING_MISSING}
