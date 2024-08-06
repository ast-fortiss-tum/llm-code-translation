"""
We used a simple heuristic to clean the generations of the open-source models. Feel free to play with the extraction
heuristic to get better results.
"""

import json
import os
import re
import argparse
import logging
from typing import List, LiteralString

from codetrans.postprocessing import Postprocessor
from codetrans.utils import copy_file
from codetransbench.utils.cli_abstraction import CLIArgumentsCleanGenerations

from codetransbench.utils.config import Config, load_config


logger = logging.getLogger(__name__)


def list_files(startpath) -> list[LiteralString | str | bytes]:
    files = []
    for root, dirs, walkfiles in os.walk(startpath):
        for name in walkfiles:
            if ".yml" not in name:
                files.append(os.path.join(root, name))

    return files


def clean_codetrans_transations(dataset: str, model_name: str, template_type: str, attempt: int, config: Config):
    main_path = config.output_dir / f"{model_name}_{template_type}" / f"iteration_{attempt}" / dataset
    output_path = config.output_dir / config.cleaned_dir_name / model_name / template_type / f"iteration_{attempt}"

    files = list_files(main_path)
    # print(files)

    postprocessing_report = {}

    for f in files:

        splitted = f.split(os.path.sep)
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, "r") as file:
            data = file.read()

        # remove the leading whitespace from beginning of the file
        data = data.lstrip()

        data = data.replace("</s>", "")
        data = data.replace("<|endoftext|>", "")
        # dolphin-2.6-phi-2
        data = data.replace("<|im_end|>", "")
        # Llama 3
        data = data.replace("<|eot_id|>", "")
        # Phi-3
        data = data.replace("<|end|>", "")

        if "# Token size exceeded" != data.strip():

            pp = Postprocessor(data, target_lang)

            pp.postprocessing_chain_complex()

            data = pp.text

            java_errors = []
            if target_lang == "Java":
                data, java_errors = replace_main_java_class_name_with_filename(filename, data)

            postprocessing_report[f"{dataset}_{source_lang}_{target_lang}_{filename}"] = {
                "pp_steps": [el.name for el in pp.report],
                "pp_errors": (
                    [el.name for el in pp.error_report] + java_errors
                    if java_errors
                    else [el.name for el in pp.error_report]
                ),
            }

            if target_lang == "Java" and dataset == "evalplus":
                data = "package com.example;\n" + data

        write_path = output_path / dataset / source_lang / target_lang
        os.makedirs(write_path, exist_ok=True)
        with open(write_path / filename, "w") as file:
            file.write(data)

    # save the metadata next to the cleaned data
    metadata = main_path / "metadata.yml"
    if os.path.exists(metadata):
        copy_file(metadata, output_path)

    pp_report_dir = config.postprocessing_reports_dir / f"{model_name}_{template_type}_{attempt:02d}"
    os.makedirs(pp_report_dir, exist_ok=True)
    with open(pp_report_dir / f"pp_report_{dataset}.json", "w") as f:
        json.dump(postprocessing_report, f)


def clean_naive_transations(dataset: str, model_name: str, template_type: str, attempt: int, config: Config):
    main_path = config.output_dir / f"{model_name}_{template_type}" / f"iteration_{attempt}" / dataset
    output_path = config.output_dir / config.cleaned_dir_name / model_name / template_type / f"iteration_{attempt}"

    files = list_files(main_path)
    # print(files)

    for f in files:

        splitted = f.split(os.path.sep)
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, "r") as file:
            data = file.read()

        # remove the leading whitespace from beginning of the file
        data = data.lstrip()

        data = data.replace("</s>", "")
        data = data.replace("<|endoftext|>", "")
        # dolphin-2.6-phi-2
        data = data.replace("<|im_end|>", "")
        # Llama 3
        data = data.replace("<|eot_id|>", "")
        # Phi-3
        data = data.replace("<|end|>", "")

        if "# Token size exceeded" != data.strip():
            pp = Postprocessor(data, target_lang)
            pp.postprocessing_chain_naive()
            data = pp.text

            java_errors = []
            if target_lang == "Java":
                data, java_errors = replace_main_java_class_name_with_filename(filename, data)

            if target_lang == "Java" and dataset == "evalplus":
                data = "package com.example;\n" + data

        write_path = output_path / dataset / source_lang / target_lang
        os.makedirs(write_path, exist_ok=True)
        with open(write_path / filename, "w") as file:
            file.write(data)


def clean_controlled_transations(dataset: str, model_name: str, template_type: str, attempt: int, config: Config):
    main_path = config.output_dir / f"{model_name}_{template_type}" / f"iteration_{attempt}" / dataset
    output_path = config.output_dir / config.cleaned_dir_name / model_name / template_type / f"iteration_{attempt}"

    files = list_files(main_path)
    # print(files)

    for f in files:

        splitted = f.split(os.path.sep)
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, "r") as file:
            data = file.read()

        # remove the leading whitespace from beginning of the file
        data = data.lstrip()

        data = data.replace("</s>", "")
        data = data.replace("<|endoftext|>", "")
        # dolphin-2.6-phi-2
        data = data.replace("<|im_end|>", "")
        # Llama 3
        data = data.replace("<|eot_id|>", "")
        # Phi-3
        data = data.replace("<|end|>", "")

        if "# Token size exceeded" != data.strip():

            pp = Postprocessor(data, target_lang)
            pp.postprocessing_chain_control()

            data = pp.text

            java_errors = []
            if target_lang == "Java":
                data, java_errors = replace_main_java_class_name_with_filename(filename, data)

            if target_lang == "Java" and dataset == "evalplus":
                data = "package com.example;\n" + data

        write_path = output_path / dataset / source_lang / target_lang
        os.makedirs(write_path, exist_ok=True)
        with open(write_path / filename, "w") as file:
            file.write(data)

    # save the metadata next to the cleaned data
    metadata = main_path / "metadata.yml"
    if os.path.exists(metadata):
        copy_file(metadata, output_path)


def clean_remove_md_for_transations(dataset: str, model_name: str, template_type: str, attempt: int, config: Config):
    main_path = config.output_dir / f"{model_name}_{template_type}" / f"iteration_{attempt}" / dataset
    output_path = config.output_dir / config.cleaned_dir_name / model_name / template_type / f"iteration_{attempt}"

    files = list_files(main_path)
    # print(files)

    for f in files:

        splitted = f.split(os.path.sep)
        filename = splitted[-1].strip()
        target_lang = splitted[-2].strip()
        source_lang = splitted[-3].strip()

        with open(f, "r") as file:
            data = file.read()

        # remove the leading whitespace from beginning of the file
        data = data.lstrip()

        data = data.replace("</s>", "")
        data = data.replace("<|endoftext|>", "")
        # dolphin-2.6-phi-2
        data = data.replace("<|im_end|>", "")
        # Llama 3
        data = data.replace("<|eot_id|>", "")
        # Phi-3
        data = data.replace("<|end|>", "")

        if "# Token size exceeded" != data.strip():
            pp = Postprocessor(data, target_lang)
            pp.postprocessing_chain_remove_md()

            data = pp.text

            java_errors = []
            if target_lang == "Java":
                data, java_errors = replace_main_java_class_name_with_filename(filename, data)

            if target_lang == "Java" and dataset == "evalplus":
                data = "package com.example;\n" + data

        write_path = output_path / dataset / source_lang / target_lang
        os.makedirs(write_path, exist_ok=True)
        with open(write_path / filename, "w") as file:
            file.write(data)

    # save the metadata next to the cleaned data
    metadata = main_path / "metadata.yml"
    if os.path.exists(metadata):
        copy_file(metadata, output_path)


# only the Java source files need clean up to test for faulty files.
def clean_source(dataset: str, config: Config):
    main_path = config.dataset_dir / dataset / "Java" / "Code"
    output_path = config.dataset_dir / dataset / "Java" / "Cleaned"

    files = list_files(main_path)
    # print(files)

    for f in files:

        splitted = f.split(os.path.sep)
        filename = splitted[-1].strip()
        target_lang = "Java"
        source_lang = "Java"

        with open(f, "r", errors="ignore") as file:
            data = file.read()

        if target_lang == "Java":
            data, _ = replace_main_java_class_name_with_filename(filename, data)

        if target_lang == "Java" and dataset == "evalplus":
            data = "package com.example;\n" + data

        write_path = output_path
        os.makedirs(write_path, exist_ok=True)
        with open(write_path / filename, "w") as file:
            file.write(data)


def replace_main_java_class_name_with_filename(filename: str, data: str) -> tuple[str, list[str]]:
    data_new = re.sub(
        r"public\s*class\s*[^{]+{",
        "public class " + filename.split(".")[0] + " {",
        data,
    )
    match = re.search(r"public\s*class\s*([^{]+){", data)
    if data_new != data:
        if match is not None:
            old_class_name = match.groups()[0].strip()
            parts = re.split(r"\s+", old_class_name)
            if len(parts) > 1:
                # example: class Main implements Runnable
                old_class_name = parts[0]
                remaining = " ".join(parts[1:])
                data_new = data_new.replace(filename.split(".")[0], filename.split(".")[0] + " " + remaining)
                logger.info(f"added {remaining} to class in {filename}")
            data = data_new.replace(old_class_name, filename.split(".")[0])
            if data != data_new:
                logger.info(f"Replaced instances of {old_class_name} with filename in Java file {filename}")
    elif "public class" in data:
        logger.debug(f"No need to replace the main public class for {filename}")
    else:
        # do not replace private or static class definitions (use 'rivate' to have same character length as static for the negative lookbehind)
        data = re.sub(
            r"(?<!rivate|static)\s+class\s*Main\s*{",
            "public class " + filename.split(".")[0] + " {",
            data,
        )
        if data_new != data:
            data = data.replace("Main", filename.split(".")[0])
        if data != data_new:
            logger.info(f"Replaced instances and definition of class Main with filename in Java file {filename}")
        else:
            logger.warning(f"No public class or class Main in Java file {filename}")
            return data, ["NO_PUBLIC_OR_MAIN_CLASS"]
    return data, []


def clean_generations(args: CLIArgumentsCleanGenerations, config: Config):

    if args.template_type == "test_source":
        clean_source(args.dataset, config)
    elif args.pp_type == "remove_md":
        clean_remove_md_for_transations(args.dataset, args.model, args.template_type, args.attempt, config)
    elif args.pp_type == "naive":
        clean_naive_transations(args.dataset, args.model, args.template_type, args.attempt, config)
    elif args.pp_type == "controlled":
        clean_controlled_transations(args.dataset, args.model, args.template_type, args.attempt, config)
    elif ("llamafile_" in args.model, config) or ("ollama" in args.model) or ("langchain" in args.model):
        clean_codetrans_transations(args.dataset, args.model, args.template_type, args.attempt, config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="clean open-source model generations given a dataset and a model")
    parser.add_argument(
        "--model",
        help="model to use for code translation.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="dataset to use for code translation.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--template_type",
        help="type of the prompt template to use for code translation. ",
        required=True,
        type=str,
    )
    parser.add_argument("--attempt", help="Attempt number to clean.", required=True, type=int)
    parser.add_argument(
        "--pp_type",
        help="type of the post processing chain to use for output cleanup. ",
        required=True,
        type=str,
    )

    config = load_config()

    # nsp = CLIArgumentsCleanGenerations()
    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsCleanGenerations(**vars(parser.parse_args()))

    try:
        clean_generations(args, config)
    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
