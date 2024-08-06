import argparse
import logging
from codetrans.utils import remove_file_or_directory

from codetransbench.execution.compile_and_test_dataset import IMPLEMENTED_PLS
from codetransbench.execution.compile_avatar import CompileAndTestAvatar
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config, load_config


logger = logging.getLogger(__name__)


class CompileAndTestBasicbench(CompileAndTestAvatar):
    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
        self.dataset = args.dataset


def main(args: CLIArgumentsExecution, config: Config):
    logger.info("testing translations")
    if args.target_lang not in IMPLEMENTED_PLS:
        logger.info(
            f"language: {args.target_lang} is not yet supported. select from the following languages {str(IMPLEMENTED_PLS)}"
        )
        return

    ctb = CompileAndTestBasicbench(args, config)
    ctb.setup_files()
    ctb.compile_and_test_generated_files()
    ctb.write_reports()
    remove_file_or_directory(ctb.temp_dir.parent, raise_value_error=False)
    logger.info("finished testing translations")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execute basicbench tests")
    parser.add_argument(
        "--source_lang",
        help="source language to use for code translation.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--target_lang",
        help="target language to use for code translation.",
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
    parser.add_argument("--attempt", help="attempt number", required=True, type=int)

    # nsp = CLIArgumentsExecution()
    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsExecution(**vars(parser.parse_args()))
    config = load_config()
    try:
        main(args, config)
    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
