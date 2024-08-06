import logging
import re
from codetrans.postprocessing import comment_syntax

logger = logging.getLogger(__name__)


def extract_error_messages(output: str):
    """Extract error messages from dotnet msbuild compiler."""
    lines = output.split("\n")
    errors = []
    warnings = []
    for i, line in enumerate(lines):
        if "error CS" in line:
            error_pattern = r"(?:.*:) (error CS[\d]+:.*)(?: \[.*\.csproj])"
            warnings_pattern = r"(?:.*:) (warning CS[\d]+:.*)(?: \[.*\.csproj])"
            match = re.match(error_pattern, line)
            if match:
                errors.append(match.group(1))
            match = re.match(warnings_pattern, line)
            if match:
                warnings.append(match.group(1))
    return errors


def extract_error_messages_with_line(output: str):
    """Extract error messages from dotnet msbuild compiler."""
    lines = output.split("\n")
    errors = []
    warnings = []
    for i, line in enumerate(lines):
        if "error CS" in line:
            error_pattern = r"(?:.+cs\()([\d,]+)\): (error CS[\d]+:.*)(?: \[.*\.csproj])"
            warnings_pattern = r"(?:.+cs\()([\d,]+)\): (warning CS[\d]+:.*)(?: \[.*\.csproj])"
            match = re.match(error_pattern, line)
            if match:
                errors.append(f"in line ({match.group(1)}): {match.group(2)}")
            else:
                error_pattern = r"(?:.*:) (error CS[\d]+:.*)(?: \[.*\.csproj])"
                match = re.match(error_pattern, line)
                if match:
                    errors.append(match.group(1))
            match = re.match(warnings_pattern, line)
            if match:
                warnings.append(f"in line ({match.group(1)}): {match.group(2)}")
            else:
                warnings_pattern = r"(?:.*:) (warning CS[\d]+:.*)(?: \[.*\.csproj])"
                match = re.match(warnings_pattern, line)
                if match:
                    warnings.append(match.group(1))
    if not errors:
        logger.warning("no errors detected")
    return errors


def extract_line_from_code(code: str, line_number: int):
    lines = code.splitlines()
    # lines in error messages are 1-indexed
    return lines[line_number + 1]


def tag_line_in_code(code: str, line_number: int, language: str, message: str):
    lines = code.splitlines()
    # lines in error messages are 1-indexed
    lines[line_number + 1] += f"{comment_syntax(language)[0]} {message}"
    return "\n".join(lines)
