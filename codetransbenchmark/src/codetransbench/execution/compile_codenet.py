import logging
import argparse
import subprocess
import time
from codetrans.utils import remove_file_or_directory

from codetransbench.execution.compile_and_test_dataset import (
    CompileAndTest,
    IMPLEMENTED_PLS,
)
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config, load_config
from codetransbench.utils.string_compare import string_equality_ignoring_newline

logger = logging.getLogger(__name__)


class CompileAndTestCodeNet(CompileAndTest):

    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
        self.dataset = "codenet"

    def read_test_input_output(self, file_index: int) -> tuple[str, str]:
        with open(
            self.test_dir / (self.file_name_without_extension(file_index) + "_in.txt"),
            "r",
        ) as f:
            f_in = f.read()
        with open(
            self.test_dir / (self.file_name_without_extension(file_index) + "_out.txt"),
            "r",
        ) as f:
            f_out = f.read()
        return f_in, f_out


def main(args: CLIArgumentsExecution, config: Config):
    logger.info("testing translations")

    if args.target_lang not in IMPLEMENTED_PLS:
        logger.info(
            f"language: {args.target_lang} is not yet supported. select from the following languages {str(IMPLEMENTED_PLS)}"
        )
        return

    ctc = CompileAndTestCodeNet(args, config)
    ctc.setup_files()
    ctc.compile_and_test_generated_files()
    ctc.write_reports()
    remove_file_or_directory(ctc.temp_dir.parent, raise_value_error=False)

    logger.info("finished testing translations")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execute codenet tests")
    parser.add_argument(
        "--source_lang",
        help="source language to use for code translation. ",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--target_lang",
        help="target language to use for code translation. ",
        required=True,
        type=str,
    )
    parser.add_argument("--model", help="model to use for code translation.", required=True, type=str)
    parser.add_argument(
        "--report_dir",
        help="path to directory to store report",
        required=True,
        type=str,
    )
    parser.add_argument("--attempt", help="Attempt number to test.", required=True, type=int)

    # nsp = CLIArgumentsExecution()
    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsExecution(**vars(parser.parse_args()))
    config = load_config()
    try:
        main(args, config)
    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
