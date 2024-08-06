import logging
import argparse
import os
from pathlib import Path
import subprocess
import time
from codetrans.utils import remove_file_or_directory

from codetransbench.execution.compile_and_test_dataset import (
    CompileAndTest,
    IMPLEMENTED_PLS,
)
from codetransbench.execution.compile_codenet import CompileAndTestCodeNet
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config, load_config
from codetransbench.utils.string_compare import string_equality_ignoring_newline

logger = logging.getLogger(__name__)


class CompileAndTestCodeNetSource(CompileAndTestCodeNet):

    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
        self.dataset = "codenet"

    def setup_files(self):
        # This is the directory of the source files of the dataset
        if self.args.source_lang == "Java":
            self.translation_dir = self.config.dataset_dir / self.dataset / self.args.source_lang / "Cleaned"
        else:
            self.translation_dir = self.config.dataset_dir / self.dataset / self.args.source_lang / "Code"
        self.test_dir = self.config.dataset_dir / self.dataset / self.args.source_lang / "TestCases"
        os.makedirs(
            Path(self.report_dir) / self.args.model / self.args.template_type / f"iteration_{self.args.attempt}",
            exist_ok=True,
        )
        self.files = [f for f in os.listdir(self.translation_dir) if f != ".DS_Store"]
        os.makedirs(self.temp_dir, exist_ok=True)


def main(args: CLIArgumentsExecution, config: Config):
    logger.info("testing source code")

    if args.target_lang not in IMPLEMENTED_PLS:
        logger.info(
            f"language: {args.target_lang} is not yet supported. select from the following languages {str(IMPLEMENTED_PLS)}"
        )
        return

    ctc = CompileAndTestCodeNetSource(args, config)
    ctc.setup_files()
    ctc.compile_and_test_generated_files()
    ctc.write_reports()
    remove_file_or_directory(ctc.temp_dir.parent, raise_value_error=False)

    logger.info("finished testing source code")


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

    # nsp = CLIArgumentsExecution()
    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsExecution(**vars(parser.parse_args()))
    config = load_config()
    try:
        main(args, config)
    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
