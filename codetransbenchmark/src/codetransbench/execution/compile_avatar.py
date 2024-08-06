import logging
import os
from subprocess import Popen
import argparse
import subprocess
import time
from codetrans.utils import remove_file_or_directory

from codetransbench.execution.compile_and_test_dataset import (
    CompileAndTest,
    IMPLEMENTED_PLS,
    get_filenames_from,
)
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config, load_config
from codetransbench.utils.string_compare import string_equality_ignoring_newline

logger = logging.getLogger(__name__)

MAX_INFINITE_LOOPS = 4


class CompileAndTestAvatar(CompileAndTest):

    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
        self.dataset = "avatar"

    def compile_and_test_generated_files(self):
        for i in range(len(self.files)):

            try:
                logger.info("Filename: " + self.files[i])
                if "# Token size exceeded" in open(self.translation_dir / self.files[i], "r", errors="ignore").read():
                    self.token_exceeded.append(self.files[i])

                self.compile_file(self.args.target_lang, file_index=i)

                tests_passed = 0
                self.infinite_loop_count = 0
                for j in range(1000):

                    if not os.path.exists(self.test_dir / (self.file_name_without_extension(i) + f"_{j}.in")):
                        # one index over the number of tests for the file
                        if tests_passed == j:
                            self.test_passed.append(self.files[i])
                        break

                    if ((self.infinite_loop_count >= MAX_INFINITE_LOOPS) and (tests_passed == 0)) or (
                        self.infinite_loop_count >= MAX_INFINITE_LOOPS + 2
                    ):
                        logger.info(
                            f"The maximum number of infinite loops / timeouts ({MAX_INFINITE_LOOPS}) during tesing is reached. Skip the remaining test cases."
                        )
                        break

                    print("Test case", j)

                    f_in, f_out = self.read_test_input_output(i, j)

                    p = self.run_file_or_binary(self.args.target_lang, file_index=i)

                    tests_passed = self.evaluate_execution(i, f_in, f_out, p, j, tests_passed)

            except subprocess.CalledProcessError as exc:
                if "# Token size exceeded." in open(self.translation_dir / self.files[i], "r").read():
                    self.token_exceeded.append(self.files[i])
                else:
                    self.compile_failed.append((self.files[i], exc.stderr.decode()))

            except Exception as e:
                logger.error(f"Developer error: {e}")
                e.with_traceback()

            self.cleanup_generated_files_of_compilation(self.args.target_lang, file_index=i)

    def read_test_input_output(self, file_index: int, test_index: int) -> tuple[str, str]:
        with open(
            self.test_dir / (self.file_name_without_extension(file_index) + f"_{test_index}.in"),
            "r",
        ) as f:
            f_in = f.read()
        with open(
            self.test_dir / (self.file_name_without_extension(file_index) + f"_{test_index}.out"),
            "r",
        ) as f:
            f_out = f.read()
        return f_in, f_out

    def evaluate_execution(
        self,
        file_index: int,
        f_in: str,
        f_out: str,
        p: Popen[bytes] | None,
        test_index: int,
        tests_passed: int,
    ):
        execution_results = self.collect_execution_results(file_index, f_in, p)
        if execution_results is None:
            self.infinite_loop_count += 1
            logger.info("No execution results. Timeout expired --> Infinite loop detected.")
            return tests_passed
        stdout, stderr_data = execution_results

        f_out, stdout = self.normalize_test_values(f_out, stdout)

        if string_equality_ignoring_newline(stdout.strip(), f_out.strip()):
            tests_passed += 1
        else:
            rt_failed_file_names = get_filenames_from(self.runtime_failed)
            test_failed_file_names = get_filenames_from(self.test_failed)
            if stderr_data.decode() == "":

                if (
                    self.files[file_index] not in rt_failed_file_names
                    and self.files[file_index] not in test_failed_file_names
                ):
                    self.test_failed.append((self.files[file_index], f_in, f_out, stdout))
            else:
                if (
                    self.files[file_index] not in test_failed_file_names
                    and self.files[file_index] not in rt_failed_file_names
                ):
                    self.runtime_failed.append((self.files[file_index], stderr_data.decode()))

        return tests_passed

    def normalize_test_values(self, f_out: str, stdout: bytes):
        try:
            if float(stdout.decode(errors="ignore")) % 1 == 0:
                stdout = str(int(float(stdout.decode(errors="ignore"))))
                f_out = str(int(float(f_out)))
            else:
                # find how many decimal points are there in the output
                stdout_temp = stdout.decode(errors="ignore").strip()
                f_out_temp = f_out.strip()
                f_out_total_dec_points = len(f_out_temp.split(".")[1])
                stdout_total_dec_points = len(stdout_temp.split(".")[1])
                min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                stdout = str(round(float(stdout.decode(errors="ignore")), min_dec_points))
                f_out = str(round(float(f_out), min_dec_points))

        except:
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="ignore")
        return f_out, stdout


def main(args: CLIArgumentsExecution, config: Config):
    logger.info("testing translations")

    if args.target_lang not in IMPLEMENTED_PLS:
        logger.info(
            f"language: {args.target_lang} is not yet supported. select from the following languages {str(IMPLEMENTED_PLS)}"
        )
        return

    cta = CompileAndTestAvatar(args, config)
    cta.setup_files()
    cta.compile_and_test_generated_files()
    cta.write_reports()
    remove_file_or_directory(cta.temp_dir.parent, raise_value_error=False)
    logger.info("finished testing translations")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execute avatar tests")
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
