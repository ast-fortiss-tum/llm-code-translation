import itertools


def language_pairs(dataset: str):

    languages = ["Python", "Java", "Go", "Rust", "C#"]  # "C", "C++"

    valid_permutations = itertools.permutations(languages, 2)
    valid_permutations_tmp = set()
    match dataset:
        case "avatar":
            for pair in valid_permutations:
                if pair[0] in ["Python", "Java"]:
                    valid_permutations_tmp.add(pair)
        case "codenet":
            for pair in valid_permutations:
                if pair[0] in ["Python", "Java", "Go"]:  # "C", "C++":
                    valid_permutations_tmp.add(pair)
        case "evalplus":
            valid_permutations_tmp = [("Python", "Java")]
        case "basicbench":
            for pair in valid_permutations:
                if pair[0] in ["Python"]:
                    valid_permutations_tmp.add(pair)
        case "bithacks":
            for pair in valid_permutations:
                if pair[0] in ["Python"]:
                    valid_permutations_tmp.add(pair)
    valid_permutations = valid_permutations_tmp

    return valid_permutations
