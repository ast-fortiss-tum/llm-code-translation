import glob
import logging
import os
import argparse
from pathlib import Path
import subprocess
from codetransbench.execution.compile_and_test_dataset import IMPLEMENTED_PLS, CompileAndTest, ExtendedCompileAndTest
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution

from codetrans.utils import remove_file_or_directory, copy_file, remove_contents_of_directory

from codetransbench.utils.config import Config, load_config

logger = logging.getLogger(__name__)


class CompileAndTestEvalPlus(CompileAndTest):

    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
        self.dataset = "evalplus"
        self.token_exceeded = []

    def setup_files(self):
        super().setup_files()
        self.test_dir = self.config.dataset_dir / self.dataset / "evalplus_java"

        self.files_src_dir = self.test_dir / "src" / "main" / "java" / "com" / "example"
        self.files_test_dir = self.test_dir / "src" / "test" / "java" / "com" / "example"

        self.ordered_files = sorted([os.path.basename(f).removesuffix(".java") for f in self.files])

    def compile_and_test_generated_files(self):
        self.setup_mvn_project()
        # self.run_mvn_tests()

        for i, f in enumerate(self.ordered_files):
            remove_contents_of_directory(self.files_src_dir)
            remove_contents_of_directory(self.files_test_dir)

            fname = f + ".java"
            test_name = f + "Test.java"
            copy_file(self.translation_dir / fname, self.files_src_dir / fname)
            copy_file(self.test_dir / "TestCases" / test_name, self.files_test_dir / test_name)

            # TODO hier weiter
            mvn_test_command = f"mvn test -Dtest={f}Test"
            logger.info(mvn_test_command)
            print(mvn_test_command)
            subprocess.run(mvn_test_command, cwd=self.test_dir, capture_output=False, shell=True)

            surefire_report_path = self.test_dir / "target" / "surefire-reports" / f"com.example.{f}Test.txt"

            if os.path.exists(surefire_report_path):
                with open(surefire_report_path, "r") as report:
                    content = report.read()
                    if "test timed out after" in content or "TestTimedOutException" in content:
                        self.runtime_failed.append((fname, "the program enters an infinite loop"))
                    elif "Errors: 0" not in content:
                        # Surefire reports runtime errors as Error: Number
                        self.runtime_failed.append((fname, content))
                    elif "Failures: 0" not in content:
                        self.test_failed.append((fname, content))
                    else:
                        self.test_passed.append(fname)
                print("Compiled", fname)
            else:
                # There is no report as the Java file could not be compiled
                f_path = self.files_src_dir / fname
                compile_command = f"javac {f_path}"
                try:
                    subprocess.run(compile_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    print(i)
                except subprocess.CalledProcessError as e:
                    if "# Token size exceeded." in open(f_path, "r").read():
                        self.token_exceeded.append(fname)
                    else:
                        self.compile_failed.append((fname, e.stderr.decode("utf-8")))

                self.cleanup_generated_files_of_compilation()

    def cleanup_generated_files_of_compilation(self):
        files_to_remove = glob.glob(str(self.files_src_dir) + "/*.class")
        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)

    def setup_mvn_project(self):
        logger.info("mvn clean install")
        print("mvn clean install")
        subprocess.run(
            "mvn clean install",
            cwd=self.test_dir,
            capture_output=False,
            shell=True,
        )

    def run_mvn_tests(self):
        logger.info("mvn test")
        subprocess.run(
            "mvn test",
            cwd=self.test_dir,
            capture_output=False,
            shell=True,
        )

    def write_reports(self):
        super().write_reports()
        ExtendedCompileAndTest.write_reports(self)


def main(args: CLIArgumentsExecution, config: Config):
    # BUG this has to be fixed
    logger.warning("This is not working yet")
    return

    logger.info("testing translations")

    if args.target_lang != "Java":
        logger.info(f"language: {args.target_lang} is not yet supported. select from the following languages: Java")
        return

    cte = CompileAndTestEvalPlus(args, config)
    cte.setup_files()
    cte.compile_and_test_generated_files()
    cte.write_reports()

    logger.info("finished testing translations")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="execute evalplus tests")
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
    parser.add_argument("--report_dir", help="path to directory to store report", required=True, type=str)
    parser.add_argument("--attempt", help="attempt number", required=True, type=int)
    parser.add_argument("--template_type", required=True, type=str)
    parser.add_argument("--dataset", default="evalplus", type=str)
    

    # nsp = CLIArgumentsExecution()
    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsExecution(**vars(parser.parse_args()))
    config = load_config()
    try:
        main(args, config)
    except Exception as e:
        logger.exception(e)
        logger.warning("Something went wrong here")
