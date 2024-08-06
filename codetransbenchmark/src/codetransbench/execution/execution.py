import logging

from codetransbench.execution.compile_codenet import main as codenet_main
from codetransbench.execution.compile_avatar import main as avatar_main
from codetransbench.execution.compile_codenet_source import main as codenet_source_main
from codetransbench.execution.compile_avatar_source import main as avatar_source_main
from codetransbench.execution.compile_evalplus import main as evalplus_main
from codetransbench.execution.compile_basicbench import main as basicbench_main
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config

logger = logging.getLogger(__name__)


def execute_translations(args: CLIArgumentsExecution, config: Config):
    logger.info("start executing translations tests")

    try:
        match args.dataset:
            case "codenet":
                codenet_main(args, config)
            case "avatar":
                avatar_main(args, config)
            case "evalplus":
                evalplus_main(args, config)
            case "basicbench" | "bithacks":
                basicbench_main(args, config)
            case _:
                logger.info(
                    f"dataset: {args.dataset} is not yet supported. select from the following datasets [avatar, codenet, evalplus, basicbench]"
                )

    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")


def execute_source_tests(args: CLIArgumentsExecution, config: Config):
    logger.info("start executing source code tests")

    try:
        match args.dataset:
            case "codenet":
                codenet_source_main(args, config)
            case "avatar":
                avatar_source_main(args, config)
            case "evalplus":
                # not needed as the Python HUmaneval evalplus is verified
                pass
            case "basicbench" | "bithacks":
                # not needed as the bithacks dataset was externally verified & basicbench is a mix of others
                pass
            case _:
                logger.info(
                    f"dataset: {args.dataset} is not yet supported. select from the following datasets [avatar, codenet, evalplus, basicbench]"
                )

    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
