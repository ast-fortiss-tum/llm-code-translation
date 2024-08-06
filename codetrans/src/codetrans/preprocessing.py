from pathlib import Path

from codetrans import remove_non_ISO_8859_1_characters_in_comments, remove_non_ASCII_characters_in_comments
from codetrans.dataset_interaction import possible_programming_languages


def compare_files_with_sanitised_version(path: Path, language: str):

    extensions = {"Python": "py", "C": "c", "C++": "c++", "Java": "java", "Go": "go", "Rust": "rs", "C#": "cs"}
    file_ending = "." + extensions[language]

    # check if the path is a directory
    if not path.is_dir():
        return

    # get all subdirectories and files in this directory
    files = list(path.rglob("*" + file_ending))
    # print(files)

    # sanitize each file's content
    sanitized_files_ascii = set()
    sanitized_files_ISO_8859_1 = set()
    sanitized_files_diff = set()
    for f in files:
        with open(str(f), "r", encoding="UTF-8", errors="ignore") as file:
            content = file.read()
            sanitized_content_ascii = remove_non_ASCII_characters_in_comments(content, language)
            if sanitized_content_ascii != content:
                sanitized_files_ascii.add(str(f))
            sanitized_content_ISO_8859_1 = remove_non_ISO_8859_1_characters_in_comments(content, language)
            if sanitized_content_ISO_8859_1 != content:
                sanitized_files_ISO_8859_1.add(str(f))
            if sanitized_content_ascii != sanitized_content_ISO_8859_1 and (
                sanitized_content_ascii != content or sanitized_content_ISO_8859_1 != content
            ):
                sanitized_files_diff.add(str(f))

    # compare the number of changed files in each subdirectory to the total number of changed files
    num_changed = len(sanitized_files_ascii)
    num_total = len(files)
    ratio_ascii = num_changed / num_total if num_total > 0 else 0
    num_changed = len(sanitized_files_ISO_8859_1)
    ratio_iso_8859_1 = num_changed / num_total if num_total > 0 else 0

    # print the results
    print(f"{path}: {len(sanitized_files_ascii):>2} sanitized files for ASCII, {ratio_ascii:.2f}")
    for file in sanitized_files_ascii:
        print(file)

    print(f"{path}: {len(sanitized_files_ISO_8859_1):>2} sanitized files for ISO-8859-1, {ratio_iso_8859_1:.2f}")
    for file in sanitized_files_ISO_8859_1:
        print(file)
    print(f"{path}: {len(sanitized_files_diff):>2} sanitized files changed with encoding")
    for file in sanitized_files_diff:
        print(file)


def check_sanitisation_lit():
    datasets = ["codenet", "avatar", "evalplus"]

    for ds in datasets:
        for language in set([l for l, _ in possible_programming_languages(ds)]):
            in_folder = Path("dataset") / ds / language / "Code"
            compare_files_with_sanitised_version(in_folder, language)


if __name__ == "__main__":

    check_sanitisation_lit()
