import logging
import regex
from collections import Counter
from pathlib import Path
from bidict import bidict
from enum import Enum


logger = logging.getLogger(__name__)

PROGRAMMING_LANGUAGES = set(["Python", "Java", "C#", "C", "C++", "Rust", "Go", "JavaScript", "Delphi"])


class TextStatus(Enum):
    CODE = 1
    CODE_IN_PL = 2
    CODE_FENCED_END = 3
    TEXT = 4
    UNCLEAR = 5
    UNBALANCED_END = 6
    NO_MARKDOWN = 7
    CODE_FENCE_IN_CODE = 8
    EMPTY = 9


class PostprocessingStep(Enum):
    ADDITIONAL_TRANSLATIONS = 1
    MARKDOWN_CODEBLOCKS = 2
    MISSING_MD_END = 3
    MISSING_MD_START = 4
    ESCAPED_UNDERSCORES = 5
    NATURAL_TEXT = 6
    MD_IN_CODE = 7
    CODE_HEURISTIC = 8
    NO_MARKDOWN = 9


def is_code_line(text_line: str, line_number: int, language: str):
    indicators = 0
    # Semicolons at the end of a line
    if text_line.strip().endswith((";\n", ";")):
        indicators += 1

    # Comment syntax
    if text_line.strip().startswith(tuple(comment_syntax(language))):
        indicators += 1

    # Presence of curly braces, brackets
    if any(bracket in text_line for bracket in ["{", "}", "[", "]", "(", ")"]):
        indicators += 1

    # Parentheses directly following text with no space to separate it
    indicators += len(regex.findall(r"\w+\(\)?", text_line))

    # camelCase in text
    indicators += len(
        regex.findall(
            r"^[a-z][a-z0-9]*(([A-Z][a-z0-9]+)*[A-Z]?|([a-z0-9]+[A-Z])*|[A-Z])$",
            text_line,
        )
    )

    # snake_case in text
    indicators += len(regex.findall(r"([a-z]+(?:_[a-z]+)+)|([A-Z]+(?:_[A-Z]+)+)", text_line))

    # keywords in text
    indicators += sum(
        [
            1 if kw in text_line else 0
            for kw in [
                "while",
                "for",
                "if",
                "else",
                "break",
                "continue",
                "elif",
                "def",
                "func ",
                "return",
                "let ",
                "struct ",
                "loop ",
                "func ",
                "const ",
                "fn ",
                "import ",
                "var ",
                "class ",
                "public ",
                "static ",
                "void ",
                "package ",
                "namespace ",
            ]
        ]
    )

    # operators in text
    indicators += sum(
        [
            1 if op in text_line else 0
            for op in [
                "=",
                "+",
                "*",
                "&",
                "&&",
                "|",
                "||",
                "<",
                ">",
                "==",
                "!=",
                ">=",
                "<=",
                ">>",
                "<<",
                "::",
                "%",
                "!(",
            ]
        ]
    )

    # text is indented with two spaces or tab
    if regex.search(r"^(([ ][ ])|(\t))+\S", text_line):
        indicators += 1

    # ------------ Indicators of normal text ------------
    # Markdown inline code style
    indicators -= len(regex.findall(r" `[\w\(\)\[\]\.]+`", text_line)) * 2

    # line starts with bullet point
    indicators -= len(regex.findall(r"^[ ]*[*-] [A-Z][a-z]*[ ,:]", text_line)) * 5

    # line starts with enumeration (1. Word )
    indicators -= len(regex.findall(r"^[ ]*\d+\. [A-Z][a-z]*[ ,:]", text_line)) * 5

    startswith_list = [
        "Sure",
        "Here",
        "Here's",
        "Please",
        "Note",
        "In ",
        "Explanation",
        "The ",
        "This ",
        "###",
        "Output:",
        "A:\n",
        "Finally",
        "We ",
        "You ",
        "To ",
        "Instead ",
        "Next, ",
        "Otherwise, ",
        "Hence, ",
        "I have ",
        "I hope ",
        "I also ",
        "I added",
        "I used ",
        "I translated ",
        "Also ",
        "Also, ",
        "And ",
        "Let's ",
        "Let me ",
    ]
    # startswith_list += ["- " + word for word in startswith_list]
    if any(text_line.lstrip().startswith(word) for word in startswith_list):
        indicators = -20

    logger.debug(f"line {line_number}: {indicators}", "||", text_line.removesuffix("\n"))
    # print(f"line {line_number}: {indicators}", "||", text_line.removesuffix("\n"))

    # check threshold
    return indicators > 0, indicators


def extract_code_from(text: str, language: str) -> tuple[str, int]:
    code = ""
    total_score = 0
    for ix, line in enumerate(text.splitlines(keepends=True)):
        is_code, score = is_code_line(line, ix + 1, language)
        if is_code:
            code += line
            total_score += score

        # TODO maybe use total_score / num_code_lines
    return code, total_score


def code_scoring_for(text: str, language: str) -> tuple[int | float, int]:
    code = ""
    total_score = 0
    code_score = 0
    lines = text.splitlines(keepends=True)
    code_lines_count = 0
    for ix, line in enumerate(lines):
        is_code, score = is_code_line(line, ix + 1, language)
        if is_code:
            code += line
            code_score += score
            code_lines_count += 1
        total_score += score
    # TODO maybe use total_score / num_code_lines
    if len(lines) == 0 or code_lines_count == 0:
        return code_score, total_score
    return code_score / code_lines_count, total_score


def is_code(text: str, language: str) -> TextStatus:
    text = text.replace("\n\n", "\n")
    code, _ = extract_code_from(text, language)
    if code == text:
        return TextStatus.CODE
    elif code == "":
        return TextStatus.TEXT
    else:
        return TextStatus.UNCLEAR


def comment_syntax(language: str) -> list[str]:
    comment = {"Python": ["#", '"""']}
    for l in ["Java", "Go", "Rust", "C", "C++", "C#"]:
        comment[l] = ["//", "/*"]

    if language not in comment.keys():
        raise NotImplementedError(f"There is no comment syntax implemented for the programming language {language}.")

    return comment[language]


def semicolon_syntax(language: str) -> bool:
    if language == "Python":
        return False
    if language in ["Java", "Go", "Rust", "C", "C++", "C#"]:
        return True
    raise NotImplementedError(f"Whether the programming language {language} uses semicolons is not implemented.")


def fix_escaped_underscores(text: str, report_change: bool = False) -> str | tuple[str, bool]:
    # important do not change paths: path\to\__file__
    pattern = regex.compile(r"\\_(?!_)")
    fixed = regex.sub(pattern, "_", text)
    if fixed != text:
        logger.info("Fixed escaped underscores.")

    if report_change:
        return fixed, fixed != text
    return fixed


def extract_markdown_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract markdown codeblocks of the following form:
    ```language
    code_block
    code_block
    ```

    ~~~language
    code_block
    code_block
    ~~~

    The language is optional.
    """
    pattern = regex.compile(
        r"(?:`{3}([\w]*)\n([\S\s]+?\n)`{3})|(?:~{3}([\w]*)\n([\S\s]+?\n)~{3})",
        regex.DOTALL | regex.MULTILINE,
    )

    # (?:`{3}([\w]*)\n([\S\s]+?)\n`{3}(?!\w))|(?:~{3}([\w]*)\n([\S\s]+?)\n~{3}(?!\w))
    matches = pattern.finditer(text)

    result = []
    for match in matches:
        language = match.group(1)
        code_block = match.group(2)
        t = (language, code_block)
        result.append(t)
    return result


def extract_markdown_code_blocks_with_pl(text: str, language: str) -> list[str]:
    """
    Extract code blocks from Markdown text that match a specified programming language.

    Args:
        text (str): The input Markdown text to search for code blocks.
        language (str): The programming language to extract code blocks for. Supported languages are Python, Java, Rust, Go, C, C++, C#, and Delphi.

    Returns:
        List[str]: A list of code block strings that match the specified programming language.

    Raises:
        NotImplementedError: If the mapping to the Markdown tag for the given programming language is not implemented.
    """
    markdown_languages = markdown_pl_mapping_bidict()
    if language not in markdown_languages.keys():
        raise NotImplementedError(
            f"The mapping to the markdown tag for the given programming language is not implemented: {language}"
        )

    extracted_blocks = extract_markdown_code_blocks(text)

    result = [block for block_language, block in extracted_blocks if block_language == markdown_languages[language]]

    return result


def markdown_pl_mapping_bidict() -> bidict[str, str]:
    """
    A bidirectional mapping between programming languages and their corresponding Markdown code block identifiers.

    :return: A bidict object containing the mapping between markdown programming language identifiers and their corresponding code identifiers.
    """
    markdown_languages = {"C++": "cpp", "C#": "csharp"}
    for lang in ["Python", "Java", "Rust", "Go", "C", "Delphi", "YAML", "JSON"]:
        markdown_languages[lang] = lang.lower()
    return bidict(markdown_languages)


def pl_from_markdown_pl_tag(md_pl_tag: str) -> str:
    """
    Returns the programming language identified by the given markdown programming language identifier.

    :param md_pl_tag: The markdown programming language identifier.
    :return: The programming language of the given markdown code block tag.
    """
    inv_md_mapping = markdown_pl_mapping_bidict().inverse
    if md_pl_tag not in inv_md_mapping:
        raise NotImplementedError(
            f"The mapping to the programming language for the given markdown tag is not implemented: {md_pl_tag}"
        )
    return inv_md_mapping[md_pl_tag]


def extract_markdown_code_blocks_code_only(text: str) -> list[str]:
    """
    Extracts all code blocks from Markdown text, and returns a list containing only the code.

    Args:
        text (str): The input text to search for code blocks.

    Returns:
        List[str]: A list of code block strings, with each string representing a single code block.
    """
    extracted_blocks = extract_markdown_code_blocks(text)
    return [block for _, block in extracted_blocks]


def handle_incomplete_markdown_code_fences(text: str, language: str) -> str:
    # The goal is to extract all lines of source code from text.
    # there are three cases to handle:
    # 1. opening ``` is missing:
    # {CODE}\n```
    # 2. closing ``` is missing:
    # ```\n{CODE}
    # 3. Detect unbalanced code fences followed by normal text:
    # {CODE}\n```\n\n{NATURAL TEXT}

    # For each text block before or after ```\n check if the text is code or natural text with the function check_if_code
    # Then put together all code blocks and remove natural text

    # Opening and closing code fence markers

    # opening marker at the start of the text
    opening_marker = r"\A```"

    # closing marker at the end of the text
    closing_marker = r"\s*```[\s]*\Z"

    code_fence = r"(?:^`{3}[\w]*\n)|(?:^~{3}[\w]*\n)"

    # Check for unbalanced fences
    match = regex.search(opening_marker, text, flags=regex.MULTILINE)
    if not match:
        logger.info("No opening code fence found")

    blocks = regex.split(code_fence, text, flags=regex.MULTILINE)
    if len(blocks) == 1:
        logger.info("No code fence found")
    for block in blocks:
        code, score = extract_code_from(block, language)

        # if score on both sides of the fence is > 0: both code
        # if score is higher with extract_markdown_code_blocks("```{target_lang}\n" + text, target_lang) than extract_markdown_code_blocks(data, target_lang)

    # Replace all fences with empty strings
    text = regex.sub(opening_marker, "", text, flags=regex.MULTILINE)
    text = regex.sub(closing_marker, "", text, flags=regex.MULTILINE)
    return text


def delete_code_fences(text: str) -> str:
    code_fence = r"(?:^`{3}[\w]*\n)|(?:^~{3}[\w]*\n)"
    return regex.sub(code_fence, "", text, flags=regex.MULTILINE)


def repeated_lines(text: str, max_repeat: int) -> str:
    """Cuts of a text after a block of max_repeat_identical lines.

    If there is not repeated text, the original text is returned. This function helps cutting off hallucinations.

    :param text: The input text.
    :param max_repeat: Maximum number of times a line can be repeated in a row.

    :return: The text cut of after a line was repeated max_repeat_times in a row if such a line exists.
    """

    lines = text.splitlines(keepends=True)
    lines_dict = Counter(lines)

    for i, key in enumerate(lines):
        if (lines_dict[key] > max_repeat) and (key * max_repeat in "".join(lines[:i])):
            print("Cutoff at line", i)
            return "".join(lines[:i])

    return text


def postprocessing_chain_complex(text: str, language: str):
    report = []
    # sanitised = remove_additional_translations(text, language)
    # if sanitised != text:
    #     text = sanitised
    #     report.append(PostprocessingStep.ADDITIONAL_TRANSLATIONS)

    text, changed = fix_escaped_underscores(text, report_change=True)
    if changed:
        report.append(PostprocessingStep.ESCAPED_UNDERSCORES)

    # balance_markdown_code_fences(text)

    match check_text_before_first_code_marker(text, language):
        case TextStatus.NO_MARKDOWN:
            raise NotImplemented
        case TextStatus.CODE_IN_PL:
            raise NotImplementedError()

    code, score = extract_code_from(text, language)
    if code != text.replace("\n\n", "\n"):
        report.append(PostprocessingStep.CODE_HEURISTIC)
        text = code

    sanitised = delete_natural_text_from(text, language)
    if sanitised != text:
        text = sanitised
        report.append(PostprocessingStep.NATURAL_TEXT)

    return text, report


# TODO use this function to check if opening markdown tag is missing
def check_text_before_first_code_marker(text: str, language: str) -> TextStatus:
    before_first_pl_tag = text_before_first_code_marker_with_tag(text, language)
    before_first_code_tag = text_before_first_code_marker(text)

    if before_first_code_tag == text:
        # no markdown code tag (with or without language tag)
        return TextStatus.NO_MARKDOWN  # --> remove natural language or use directly
    if before_first_pl_tag == "":
        # the text begins with the opening markdown code fence with a language tag
        return TextStatus.CODE_IN_PL  # --> check if code after tag and balanced
    if before_first_pl_tag == before_first_code_tag:
        # check if code before the tag
        return is_code(before_first_code_tag)  #########
    else:
        # the markdown code fence has no language tag
        if before_first_code_tag == text.rstrip().removesuffix("```").removesuffix("~~~"):
            # code fence only at the end of the text
            status = is_code(before_first_code_tag, language)
            if status == TextStatus.CODE:
                return TextStatus.CODE_FENCED_END
            return status

        if before_first_code_tag == "":
            # check if rest / until ``` code if yes extr. markdown
            return TextStatus.EMPTY

        # TODO do something to distinguish this case from #########
        return is_code(before_first_code_tag)


def text_before_first_code_marker(text: str) -> str:
    code_fence = r"(?:^`{3})|(?:^~{3})"
    before_first_code_tag = regex.split(code_fence, text, flags=regex.MULTILINE)[0]
    return before_first_code_tag


def text_before_first_code_marker_with_tag(text: str, language: str) -> str:
    tag = markdown_pl_mapping_bidict()[language]
    code_fence_with_pl_tag = r"(?:^`{3}" + tag + r")|(?:^~{3}" + tag + r")"
    before_first_pl_tag = regex.split(code_fence_with_pl_tag, text, flags=regex.MULTILINE)[0]
    return before_first_pl_tag


def is_code_between_code_markers(text: str, language: str, tagged: bool = False):
    code_fence = r"(?:^`{3})|(?:^~{3})"

    tag = markdown_pl_mapping_bidict()[language]
    code_fence_with_pl_tag = r"(?:^`{3}" + tag + r")|(?:^~{3}" + tag + r")"
    pattern = code_fence_with_pl_tag if tagged else code_fence
    after_first_code_fence = regex.split(pattern, text, maxsplit=1, flags=regex.MULTILINE)[1]

    # check if code until next ```
    blocks = regex.split(pattern, after_first_code_fence, maxsplit=1, flags=regex.MULTILINE)
    next_block = blocks[0]
    if len(blocks) == 1 and tagged:
        if is_code(next_block, language) == TextStatus.CODE:
            # check if rest is code and add missing codefence ``` at the end
            # report: ending code fence is missing
            return True, TextStatus.UNBALANCED_END
    status = is_code(next_block, language)
    return status == TextStatus.CODE, status


def code_fence_in_code(text: str, language: str):
    code_fence = r"(?:`{3}[\w]*\s*)|(?:~{3}[\w]*\s*)"
    text_blocks = regex.split(code_fence, text, flags=regex.MULTILINE)

    text_blocks = list(filter(lambda b: b != "", text_blocks))
    # set the threshold to 2 as this means that two code fences were found
    return all(is_code(b, language) == TextStatus.CODE for b in text_blocks) and len(text_blocks) > 2


class Balance(Enum):
    OPENING_MISSING = 1
    CLOSING_MISSING = 2
    BALANCED = 3
    NESTED = 4
    NO_MARKDOWN = 5


def check_balance_of_code_fences(text: str, language: str) -> set[Balance]:
    stack = []
    report = set()
    no_markdown = True

    lines = text.splitlines(keepends=True)

    for line in lines:
        line = line.lstrip()
        # Check if the line contains a code tag
        if line.startswith("```"):
            no_markdown = False
            if regex.search(r"```[\s]+", line) or line == "```":
                if stack and (stack[-1] == "code" or stack[-1] == "tag"):
                    # pop the opening tag from the stack
                    stack.pop()
                else:
                    # add the opening tag to the stack
                    stack.append("code")
            elif regex.search(r"```[\w]+", line):
                if stack:
                    if stack[-1] == "code":
                        stack.pop()
                        stack.append("tag")
                        report.add(Balance.OPENING_MISSING)
                    else:
                        # two opening tags with a language identifier
                        return {Balance.NESTED}
                else:
                    # add the opening tag to the stack
                    stack.append("tag")

    # If we end up with an opening tag on the stack, the code tags are unbalanced
    if stack:
        if stack[-1] == "tag" or text.strip() == "```":
            report.add(Balance.CLOSING_MISSING)
        else:
            # the tag ``` could be opening or closing
            before = text_before_first_code_marker(text)
            after = regex.split(r"```[\s]+", text)[1]
            scores_before = code_scoring_for(before, language)
            scores_after = code_scoring_for(after, language)
            if (scores_before[0] > scores_after[0] or (scores_before[1] > 0 > scores_after[1])) or after.strip() == "":
                # CODE
                # ```
                # NOT CODE
                report.add(Balance.OPENING_MISSING)
            elif before.strip() == "" and (
                scores_before[0] < scores_after[0] or (scores_before[1] < 0 < scores_after[1])
            ):
                # ```
                # CODE
                # or
                # SOMETHING
                # ```
                # CODE
                report.add(Balance.CLOSING_MISSING)
            else:
                # check total scores
                if scores_before[1] > scores_after[1]:
                    report.add(Balance.OPENING_MISSING)
                else:
                    report.add(Balance.CLOSING_MISSING)
        return report
    else:
        if no_markdown:
            return {Balance.NO_MARKDOWN}
        if report:
            return report
        return {Balance.BALANCED}


def delete_natural_text_from(text: str, target_pl: str = None):
    if target_pl is not None:
        pl_not_target = PROGRAMMING_LANGUAGES.difference({target_pl})

    valid_lines = ""
    for line in text.splitlines(keepends=True):
        startswith_list = [
            "Sure",
            "Here",
            "Here's",
            "The ",
            "This ",
            "You ",
            "To ",
            "Instead ",
            "For ",
            "After ",
            "Next, ",
            "Otherwise, ",
            "Hence, ",
            "I have ",
            "I hope ",
            "I also ",
            "I added",
            "I used ",
            "I translated ",
            "Also ",
            "Also, ",
            "And ",
            "Let's ",
            "Let me ",
        ]
        if any(line.lstrip().startswith(word) for word in startswith_list):
            # skip this line
            continue
        startswith_list = [
            "Please",
            "Note",
            "In ",
            "Explanation",
            "Output:",
            "A:\n",
            "### ",
            "We ",
            "Finally",
        ]
        if target_pl is not None:
            startswith_list += [f"{pl} Code" for pl in pl_not_target] + [f"{pl} code" for pl in pl_not_target]
        if any(line.lstrip().startswith(word) for word in startswith_list):
            # skip this line and the following lines
            break
        else:
            valid_lines += line

    return valid_lines


def remove_additional_translations(text: str, target_pl: str) -> str:
    if target_pl is not None:
        pl_not_target = PROGRAMMING_LANGUAGES.difference({target_pl})
    valid_lines = []
    for line in text.split("\n"):
        if target_pl is not None:
            startswith_list = [f"{pl} Code" for pl in pl_not_target]
        if any(line.strip().startswith(word) for word in startswith_list):
            # skip this line and the following lines
            break
        else:
            valid_lines.append(line)

    return "\n".join(valid_lines)


def postprocessing_chain(text: str, language: str) -> tuple[str, list[PostprocessingStep]]:
    report = []
    sanitised = remove_additional_translations(text, language)
    if sanitised != text:
        text = sanitised
        report.append(PostprocessingStep.ADDITIONAL_TRANSLATIONS)

    text, changed = fix_escaped_underscores(text, report_change=True)
    if changed:
        report.append(PostprocessingStep.ESCAPED_UNDERSCORES)

    code, score = extract_code_from(text, language)
    if code != text.replace("\n\n", "\n"):
        report.append(PostprocessingStep.CODE_HEURISTIC)
        text = code

    sanitised = delete_natural_text_from(text, language)
    if sanitised != text:
        text = sanitised
        report.append(PostprocessingStep.NATURAL_TEXT)

    return text, report


class Postprocessor:

    def __init__(self, text: str, language: str):
        self.text = text  # current text
        self.orginal_text = text
        self.language = language
        self.md_language_tag = markdown_pl_mapping_bidict()[self.language]
        self.report = []  # report of which steps were necessary
        self.error_report = []

    def report_change(self, sanitised: str, step: PostprocessingStep):
        if sanitised != self.text:
            self.text = sanitised
            self.report.append(step)

    def balance_markdown_code_fences(self) -> str:
        balance = check_balance_of_code_fences(self.text, self.language)
        if Balance.OPENING_MISSING in balance:
            self.text = f"```{self.md_language_tag}\n" + self.text
            self.report.append(PostprocessingStep.MISSING_MD_START)
        if Balance.CLOSING_MISSING in balance:
            self.text += "\n```\n"
            self.report.append(PostprocessingStep.MISSING_MD_END)
        if Balance.NO_MARKDOWN in balance:
            self.report.append(PostprocessingStep.NO_MARKDOWN)
        if Balance.NESTED in balance:
            # nested markdown fences are only reported not fixed
            self.error_report.append(Balance.NESTED)

    def fix_escaped_underscores(self):
        # important do not change paths: path\to\__file__
        pattern = regex.compile(r"\\_(?!_)")
        fixed = regex.sub(pattern, "_", self.text)
        self.report_change(fixed, PostprocessingStep.ESCAPED_UNDERSCORES)

    def extract_markdown_code_blocks(self, ignore_md_pl: bool = True):
        code_blocks = extract_markdown_code_blocks(self.text)
        if len(code_blocks) > 0:
            data = ""
            for lang, code in code_blocks:
                if ignore_md_pl or lang == self.md_language_tag or lang == "":
                    data += code + "\n"
            # print(self.text)
            self.report_change(data, PostprocessingStep.MARKDOWN_CODEBLOCKS)

    def extract_code_via_heutristic(self):
        code, score = extract_code_from(self.text, self.language)
        if code != self.text.replace("\n\n", "\n"):
            self.text = code
            self.report.append(PostprocessingStep.CODE_HEURISTIC)

    def delete_natural_text(self):
        sanitised = delete_natural_text_from(self.text, self.language)
        self.report_change(sanitised, PostprocessingStep.NATURAL_TEXT)

    def delete_code_fences(self):
        sanitised = delete_code_fences(self.text)
        self.report_change(sanitised, PostprocessingStep.MD_IN_CODE)

    def code_fence_in_code(self):
        if code_fence_in_code(self.text, self.language):
            self.error_report.append(TextStatus.CODE_FENCE_IN_CODE)
            self.delete_code_fences()

    def postprocessing_chain_complex(self):
        self.fix_escaped_underscores()

        self.balance_markdown_code_fences()

        if Balance.NESTED in self.error_report:
            logger.info("Nested markdown code block detected.")
            self.extract_code_via_heutristic()

        if PostprocessingStep.NO_MARKDOWN in self.report:
            self.delete_natural_text()
            self.extract_code_via_heutristic()

        self.code_fence_in_code()

        self.extract_markdown_code_blocks()

        self.delete_natural_text()

    def postprocessing_chain_radical(self):
        self.fix_escaped_underscores()
        self.delete_natural_text()
        self.extract_code_via_heutristic()

    def postprocessing_chain_control(self):
        # add the markdown tag that is in the controlled or controlled_md templates
        self.text = f"```{self.md_language_tag}\n" + self.text
        self.extract_markdown_code_blocks(ignore_md_pl=False)

    def postprocessing_chain_naive(self):
        self.delete_natural_text()
        self.delete_code_fences()

    def postprocessing_chain_remove_md(self):
        self.delete_code_fences()


if __name__ == "__main__":

    file = "..\initial_experiments\human_eval_phi-2\HumanEval_33.py"
    file = "..\initial_experiments\human_eval_phi-2\HumanEval_53.py"
    file = Path("C:\home\ma_code\codetransbenchmark\output\llamafile_mixtral_LIT\codenet\Java\Python\s745776078.py")

    # with open(file, "r") as f:
    #     for ix, line in enumerate(f.readlines()):
    #         is_code_line(line, ix + 1, "Python")

    # print("###" + text_before_first_code_marker("```") + "###")

    text = ""
    with open(file, "r") as f:
        text = f.read()

    # print(text)

    pp = Postprocessor(text, "Python")

    pp.postprocessing_chain_complex()

    print(pp.report)
    print("--------------------------------")
    print(pp.text)
