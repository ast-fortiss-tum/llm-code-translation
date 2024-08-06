import logging
import os
import argparse
import time

import regex


from codetransbench.execution.execution import execute_translations, execute_source_tests
from codetransbench.translation.clean_generations import clean_generations
from codetransbench.translation.translate_open_source import main as translation_main
from codetransbench.translation.repair import main as repair_main
from codetransbench.utils.cli_abstraction import (
    CLIArguments,
    CLIArgumentsExecution,
    CLIArgumentsCleanGenerations,
    CLIArgumentsTranslation,
    CLIArgumentsTranslationIteration,
)
from codetransbench.utils.config import load_config, Config
from codetransbench.utils.dataset_information import language_pairs
from codetransbench.utils.logging_utilities import setup_logging


logger = logging.getLogger(__name__)


def logged_batch_task(
    config: Config,
    logfile: str,
    task: str,
    batch_function: callable,
    cli_args: CLIArguments,
):

    logger.info(f"Executing command: {task} {cli_args}")

    try:
        batch_function(cli_args, config)
    except Exception as e:
        logger.error(e)
    else:
        with open(logfile, "a") as log:
            log.write(f"{time.ctime()}:\n")
            log.write(f"Executed command: {task} {cli_args}\n")


def language_pairs_for_batching(
    logfile: str,
    task: str,
    model: str,
    dataset: str,
    template: str,
    attempt: int = None,
    ignore_executed_pairs=True,
):
    executed_pairs = set()
    if os.path.exists(logfile):
        with open(logfile, "r") as log:
            log_lines = log.readlines()
            for line in log_lines:
                pattern = f"Executed command: {task}"
                if (
                    pattern in line
                    and model in line
                    and dataset in line
                    and f"'{template}'" in line
                    and (not attempt or f"attempt={attempt}" in line)
                ):
                    found_source = regex.findall(
                        pattern=r"source_lang=[\"\']([^\s]*)[\"\'], target_lang",
                        string=line,
                    )
                    found_target = regex.findall(pattern=r"target_lang=[\"\']([^\s]*)[\"\'], ", string=line)
                    pair = (found_source[0], found_target[0])
                    executed_pairs.add(pair)

    logging.info(f"Already executed pairs: {executed_pairs}")

    valid_permutations = language_pairs(dataset)

    with open(logfile, "a") as log:
        sorted_permutations = sorted(valid_permutations)
        new_permutations = []
        for args in sorted_permutations:
            if ignore_executed_pairs or args not in executed_pairs:
                new_permutations.append(args)
                executed_pairs.add(args)
        return new_permutations


def iterate_task_over_dataset(
    config: Config,
    model_name: str,
    task: str,
    template_type: str,
    logfile: str,
    datasets: list[str],
    attempt: int = 1,
    ignore_executed_pairs=False,
    pp_type: str = "complex",
):
    for ds in datasets:
        match task:

            case "translate":
                for source_lang, target_lang in language_pairs_for_batching(
                    logfile,
                    task,
                    model_name,
                    ds,
                    template_type,
                    ignore_executed_pairs=ignore_executed_pairs,
                ):
                    cli_args = CLIArgumentsTranslation(
                        model_name,
                        ds,
                        template_type,
                        source_lang,
                        target_lang,
                        k=50,
                        p=0.95,
                        temperature=0.7,
                    )
                    logged_batch_task(config, logfile, task, translation_main, cli_args)

            case "clean_generations":
                cli_args = CLIArgumentsCleanGenerations(
                    model=model_name,
                    dataset=ds,
                    template_type=template_type,
                    attempt=attempt,
                    pp_type=pp_type,
                )
                logged_batch_task(config, logfile, task, clean_generations, cli_args)
            case "test":
                test_output = str(config.testresults_dir)
                for source_lang, target_lang in language_pairs_for_batching(
                    logfile,
                    task,
                    model_name,
                    ds,
                    template_type,
                    attempt,
                    ignore_executed_pairs,
                ):
                    exec_args = CLIArgumentsExecution(
                        model=model_name,
                        dataset=ds,
                        template_type=template_type,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        report_dir=test_output,
                        attempt=attempt,
                    )
                    logged_batch_task(config, logfile, task, execute_translations, exec_args)
            case "test_source":
                test_output = str(config.testresults_dir)
                for source_lang, source_lang in set(
                    [
                        (s, s)
                        for s, _ in language_pairs_for_batching(
                            logfile,
                            task,
                            model_name,
                            ds,
                            template_type,
                            attempt,
                            ignore_executed_pairs,
                        )
                    ]
                ):
                    exec_args = CLIArgumentsExecution(
                        model=model_name,
                        dataset=ds,
                        template_type=template_type,
                        source_lang=source_lang,
                        target_lang=source_lang,
                        report_dir=test_output,
                        attempt=attempt,
                    )
                    logged_batch_task(config, logfile, task, execute_source_tests, exec_args)
            case "repair":
                for source_lang, target_lang in language_pairs_for_batching(
                    logfile,
                    task,
                    model_name,
                    ds,
                    template_type,
                    attempt,
                    ignore_executed_pairs,
                ):
                    cli_args = CLIArgumentsTranslationIteration(
                        model_name,
                        ds,
                        template_type,
                        source_lang,
                        target_lang,
                        k=50,
                        p=0.95,
                        temperature=0.7,
                        attempt=attempt,
                    )
                    logged_batch_task(config, logfile, task, repair_main, cli_args)


def setup_environment_variables():
    # This is required to ensure the C# compiler outputs are in English and not in a mixtrue with the current locale of the machine
    os.environ["DOTNET_CLI_UI_LANGUAGE"] = "en"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        help="The task to perform. Valid values: 'translate', 'clean_generations', 'test', 'repair",
        type=str,
    )
    parser.add_argument("model", help="Name of the model to use.", type=str)
    parser.add_argument("template", help="Name of the prompt template to use.", type=str)
    parser.add_argument("-a", "--attempt", default=1, help="Attempt", type=int)
    parser.add_argument(
        "-opp",
        "--output_post_processing",
        help="Which post-processing chain to apply on the LLM-generated output.",
        required=False,
        type=str,
        default="complex",
    )
    parser.add_argument(
        "-mini",
        "--mini_benchmark",
        help="Whether to use the mini-benchmark instead of the datasets.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-over",
        "--overwrite",
        help="Whether to ignore the language pairs that were already executed.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="The single dataset to use.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "-e", "--engine", help="Name of the model engine to use.", required=False, type=str, default="llamafile"
    )

    args = parser.parse_args()

    model_name = args.model.lower()
    task = args.task.lower()
    template_type = args.template.lower()
    dataset = args.dataset
    model_engine = args.engine.lower()
    opp_type = args.output_post_processing

    config = load_config()

    setup_environment_variables()

    setup_logging(task=task, model=model_name, config=config)

    logfile = f"logfile_for_python_batching_{model_name}.txt"

    datasets = ["codenet", "avatar", "evalplus", "bithacks"]
    if dataset and (dataset in datasets):
        datasets = [dataset.lower()]
    if args.mini_benchmark:
        datasets = ["basicbench"]

    if task == "test_source":
        datasets = ["codenet", "avatar"]

    if task == "repair" and "evalplus" in datasets:
        # FUTURE WORK: testing evalplus is not yet supported due to dependency issues with Maven
        datasets.remove("evalplus")

    ignore_executed_pairs = args.overwrite

    if opp_type != "complex":
        # change the directories to the output postprocessing strategy
        config.testresults_dir = config.testresults_dir.parent / f"{config.testresults_dir.stem}_{opp_type}_pp"
        config.cleaned_dir_name = f"{config.cleaned_dir_name}_{opp_type}_pp"

    iterate_task_over_dataset(
        config,
        f"{model_engine}_{model_name}",
        task,
        template_type,
        logfile,
        datasets,
        args.attempt,
        ignore_executed_pairs,
        opp_type,
    )
