import itertools


def possible_programming_languages(dataset):
    languages = ["Python", "Java", "C", "C++", "Go"]  # Replace this with your list of arguments

    valid_permutations = itertools.permutations(languages, 2)
    if "avatar" in dataset:
        valid_permutations_tmp = set()
        for pair in valid_permutations:
            if pair[0] in ["Python", "Java"]:
                valid_permutations_tmp.add(pair)
        valid_permutations = valid_permutations_tmp
    elif "evalplus" in dataset:
        valid_permutations = [("Python", "Java")]

    return valid_permutations
