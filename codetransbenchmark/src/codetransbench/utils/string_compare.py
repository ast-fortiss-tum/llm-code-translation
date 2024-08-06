# This is needed to correctly compare mulitline test outputs indepent of \r\n or \n in the files
def string_equality_ignoring_newline(string_1: str, string_2: str):
    """Checks the if two strings are equal with OS independent new line separators."""
    string_1 = "\n".join(string_1.splitlines())
    string_2 = "\n".join(string_2.splitlines())
    return string_1 == string_2
