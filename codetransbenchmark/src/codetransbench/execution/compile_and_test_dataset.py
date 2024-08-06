from io import TextIOWrapper
import json
import glob
import logging
import os
import subprocess
import sys
import time
import pandas as pd
from pathlib import Path
import platform
from subprocess import Popen, PIPE
import uuid
from codetrans.llm_abstraction import on_windows_workstation
from codetrans.utils import remove_file_or_directory, copy_file

from codetransbench.execution.compiler_output_parser import extract_error_messages, extract_error_messages_with_line
from codetransbench.translation.translate_open_source import FILE_EXTENSIONS
from codetransbench.utils.cli_abstraction import CLIArgumentsExecution
from codetransbench.utils.config import Config, load_config
from codetransbench.utils.string_compare import string_equality_ignoring_newline

IMPLEMENTED_PLS = ["Python", "Java", "C", "C++", "Go", "Rust", "C#"]


logger = logging.getLogger(__name__)


class CompileAndTest:

    def __init__(self, args: CLIArgumentsExecution, config: Config):
        self.args = args
        self.config = config
        self.dataset = "dummy_dataset_name"
        self.report_dir = args.report_dir
        self.compile_failed = []
        self.test_passed = []
        self.test_failed = []
        self.runtime_failed = []
        self.infinite_loop = []
        self.files = []
        self.token_exceeded = []
        self.ordered_unsuccessful_files = []
        self.temp_dir = self.config.temp_exec_dir / str(uuid.uuid4()) / "project"

    def setup_files(self):
        self.translation_dir = (
            self.config.output_dir
            / self.config.cleaned_dir_name
            / self.args.model
            / self.args.template_type
            / f"iteration_{self.args.attempt}"
            / self.dataset
            / self.args.source_lang
            / self.args.target_lang
        )
        self.test_dir = self.config.dataset_dir / self.dataset / self.args.source_lang / "TestCases"
        os.makedirs(
            Path(self.report_dir) / self.args.model / self.args.template_type / f"iteration_{self.args.attempt}",
            exist_ok=True,
        )
        self.files = [f for f in os.listdir(self.translation_dir) if f.endswith(FILE_EXTENSIONS[self.args.target_lang])]
        os.makedirs(self.temp_dir, exist_ok=True)

    def compile_and_run(self, target_pl: str, file_index: int):

        if target_pl not in IMPLEMENTED_PLS:
            raise NotImplementedError(
                f"language:{target_pl} is not yet supported. select from the following languages {str(IMPLEMENTED_PLS)}"
            )

        self.compile_file(target_pl, file_index)
        return self.run_file_or_binary(target_pl, file_index)

    def compile_file(self, target_pl: str, file_index: int):
        """raises subprocess.CalledProcessError"""

        path_to_file = str(self.translation_dir / self.files[file_index])

        os.makedirs(self.temp_dir, exist_ok=True)

        def run_compile_process(command: str, path_to_file: str, cwd: str | None = None, timeout: int = 30):
            return subprocess.run(
                command + " " + path_to_file, cwd=cwd, check=True, capture_output=True, shell=True, timeout=timeout
            )

        match target_pl:
            case "Python":
                run_compile_process("python -m py_compile", path_to_file)
            case "Java":
                run_compile_process("javac", path_to_file)
            case "C":
                # TODO this might not work on Windows
                run_compile_process("gcc", path_to_file, self.temp_dir, timeout=10)
            case "C++":
                # TODO this might not work on Windows
                run_compile_process("g++ -o exec_output -std=c++11", path_to_file, self.temp_dir, timeout=60)
            case "Go":
                run_compile_process("go build", path_to_file, self.temp_dir)
            case "Rust":
                run_compile_process("rustc", path_to_file, self.temp_dir)
            case "C#":
                if not os.path.isdir(self.temp_dir / "obj"):
                    subprocess.run(
                        "dotnet new console --force --verbosity quiet",
                        cwd=self.temp_dir,
                        shell=True,
                        capture_output=False,
                    )
                copy_file(Path(path_to_file), self.temp_dir / "Program.cs")
                try:
                    subprocess.run(
                        "dotnet msbuild -flp1:logfile=msbuild_errors.txt -flp1:errorsonly",
                        cwd=self.temp_dir,
                        check=True,
                        capture_output=True,
                        shell=True,
                        timeout=30,
                    )
                except subprocess.CalledProcessError as cpe:
                    build_output = ""
                    if cpe.stderr is not None:
                        build_output += cpe.stderr.decode(errors="ignore").strip() + "\n"
                    if cpe.stdout is not None:
                        build_output += cpe.stdout.decode(errors="ignore").strip()
                    if build_output != "":
                        errors = extract_error_messages_with_line(build_output)
                        if errors:
                            raise subprocess.CalledProcessError(
                                cpe.returncode, "dotnet msbuild", build_output.encode(), "\n".join(errors).encode()
                            )
            case _:
                pass

    def run_file_or_binary(self, target_pl: str, file_index: int):

        path_to_file = str(self.translation_dir / self.files[file_index])

        match target_pl:
            case "Python":
                # sys.executable resolves to the python interpreter of the current venv
                return Popen(
                    [sys.executable, path_to_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            case "Java":
                return Popen(
                    ["java", self.file_name_without_extension(file_index)],
                    cwd=self.translation_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            case "C":
                # TODO this might not work on Windows
                path_to_executable = self.temp_dir / "a.out"
                while not os.path.exists(path_to_executable):
                    time.sleep(0.2)
                return Popen(
                    ["./a.out"],
                    cwd=self.temp_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            case "C++":
                # TODO this might not work on Windows
                path_to_executable = self.temp_dir / "exec_output"
                while not os.path.exists(path_to_executable):
                    time.sleep(0.2)
                return Popen(
                    ["./exec_output"],
                    cwd=self.temp_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            case "Go" | "Rust":
                path_to_executable = self.file_name_without_extension(file_index)
                if platform.system() == "Windows":
                    path_to_executable += ".exe"
                while not os.path.exists(self.temp_dir / path_to_executable):
                    time.sleep(0.2)
                return Popen(
                    [self.temp_dir / path_to_executable],
                    cwd=self.temp_dir,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            case "C#":
                if platform.system() == "Windows" or on_windows_workstation():
                    path_to_executable = str(self.temp_dir / "bin" / "Debug" / "net7.0" / "project.exe")
                while not os.path.exists(path_to_executable):
                    time.sleep(0.2)
                return Popen(
                    [path_to_executable],
                    cwd=os.getcwd(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            case _:
                pass

    def cleanup_generated_files_of_compilation(self, target_lang: str, file_index: int):
        match target_lang:
            case "Python":
                try:
                    remove_file_or_directory(self.translation_dir / "__pycache__")
                except ValueError as ve:
                    pass
                return
            case "Java":
                # remove all .class files generated
                # self.translation_dir + "/" + self.file_name_without_extension(file_index) + ".class"
                files_to_remove = glob.glob(str(self.translation_dir) + "/*.class")
            case "Go" | "Rust" | "C" | "C++":
                try:
                    remove_file_or_directory(self.temp_dir, raise_value_error=False)
                except ValueError as ve:
                    pass
                except PermissionError as e:
                    time.sleep(1)
                    try:
                        remove_file_or_directory(self.temp_dir, raise_value_error=False)
                    except PermissionError as e:
                        logger.error(e)
                        logger.warning("The Permission error will be ignored, the file will not be deleted.")
                return
            case "C#":
                try:
                    remove_file_or_directory(self.temp_dir / "bin", raise_value_error=False)
                except ValueError as ve:
                    pass
                except PermissionError as e:
                    time.sleep(1)
                    try:
                        remove_file_or_directory(self.temp_dir / "bin", raise_value_error=False)
                    except PermissionError as e:
                        logger.error(e)
                        logger.warning("The Permission error will be ignored, the file will not be deleted.")
                return
            case _:
                return

        for f in files_to_remove:
            if os.path.isfile(f):
                os.remove(f)

    def file_name_without_extension(self, file_index: int):
        return os.path.splitext(self.files[file_index])[0]

    def compile_and_test_generated_files(self):
        for i in range(len(self.files)):

            try:
                logger.info("Filename: " + self.files[i])

                if "# Token size exceeded" in open(self.translation_dir / self.files[i], "r", errors="ignore").read():
                    self.token_exceeded.append(self.files[i])

                f_in, f_out = self.read_test_input_output(i)

                p = self.compile_and_run(self.args.target_lang, file_index=i)

                self.evaluate_execution(i, f_in, f_out, p)

            except subprocess.CalledProcessError as exc:
                if "# Token size exceeded" in open(self.translation_dir / self.files[i], "r", errors="ignore").read():
                    self.token_exceeded.append(self.files[i])
                else:
                    self.compile_failed.append((self.files[i], exc.stderr.decode(errors="ignore")))

            except Exception as e:
                logger.error(f"Developer error: {e}")
                e.with_traceback()

            self.cleanup_generated_files_of_compilation(self.args.target_lang, file_index=i)

    def read_test_input_output(self, file_index: int) -> tuple[str, str]:
        with open(self.test_dir / (self.file_name_without_extension(file_index) + "_in.txt"), "r") as f:
            f_in = f.read()
        with open(self.test_dir / (self.file_name_without_extension(file_index) + "_out.txt"), "r") as f:
            f_out = f.read()
        return f_in, f_out

    def evaluate_execution(self, file_index: int, f_in: str, f_out: str, p: Popen[bytes] | None):
        execution_results = self.collect_execution_results(file_index, f_in, p)
        if execution_results is None:
            logger.info("No execution results. Timeout expired --> Infinite loop detected.")
            return
        stdout, stderr_data = execution_results

        if string_equality_ignoring_newline(stdout.decode(errors="ignore").strip(), f_out.strip()):
            self.test_passed.append(self.files[file_index])
        else:
            if stderr_data.decode() == "":
                self.test_failed.append(
                    (
                        self.files[file_index],
                        f_in,
                        f_out,
                        stdout.decode(errors="ignore"),
                    )
                )
            else:
                self.runtime_failed.append((self.files[file_index], stderr_data.decode(errors="ignore")))

    def collect_execution_results(self, file_index: int, f_in: str, p: Popen[bytes] | None) -> tuple[bytes, bytes]:
        try:
            stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
        except subprocess.TimeoutExpired:
            self.infinite_loop.append(self.files[file_index])
            logger.info("Timeout expired")
            p.kill()
            logger.info("Killed the timed out process")
            stdout, stderr_data = p.communicate()
            logger.info("Collected process output")
            return None
        return stdout, stderr_data

    def write_reports(self):
        test_failed_files = sorted(list(set(get_filenames_from(self.test_failed))))
        runtime_failed_files = sorted(list(set(get_filenames_from(self.runtime_failed))))
        compile_failed_files = sorted(list(set(get_filenames_from(self.compile_failed))))
        infinite_loop_files = sorted(list(set(self.infinite_loop)))
        test_passed = sorted(list(set(self.test_passed)))
        test_failed_and_loop_files = []

        # To avoid the total sum is higher than 100%, if an instance is in infinite_loop and test_failed at the same time, then it will be counted as test_failed
        for instance in infinite_loop_files:
            if instance in test_failed_files:
                infinite_loop_files.remove(instance)
                test_failed_and_loop_files.append(instance)

        total_instances = (
            len(test_passed)
            + len(compile_failed_files)
            + len(runtime_failed_files)
            + len(test_failed_files)
            + len(infinite_loop_files)
        )

        statistics_dict = {
            "Total Instances": total_instances,
            "Total Correct": len(test_passed),
            "Total Runtime Failed": len(runtime_failed_files),
            "Total Compilation Failed": len(compile_failed_files),
            "Total Test Failed": len(test_failed_files),
            "Total Infinite Loop": len(infinite_loop_files),
            "Total Test Failed & Infinite Loop": len(test_failed_and_loop_files),
            "Accuracy": (len(test_passed) / total_instances) * 100,
            "Runtime Rate": (len(runtime_failed_files) / total_instances) * 100,
            "Compilation Rate": (len(compile_failed_files) / total_instances) * 100,
            "Test Failed Rate": (len(test_failed_files) / total_instances) * 100,
            "Infinite Loop Rate": (len(infinite_loop_files) / total_instances) * 100,
            "Test Failed & Infinite Loop Rate": (len(test_failed_and_loop_files) / total_instances) * 100,
        }

        txt_fp = Path(self.report_dir).joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.dataset}_compileReport_from_{str(self.args.source_lang)}_to_{str(self.args.target_lang)}.txt",
        )
        with open(txt_fp, "w", encoding="utf-8") as report:
            for stat_name, value in statistics_dict.items():
                report.writelines(f"{stat_name}: {value}\n")
                if "Loop" in stat_name or "Instances" in stat_name:
                    report.writelines("\n")

            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Successfull Test Files: {} \n".format(test_passed))
            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Failed Test Files: {} \n".format(test_failed_files))
            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Runtime Error Files: {} \n".format(runtime_failed_files))
            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Compilation Error Files: {} \n".format(compile_failed_files))
            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Infinite Loop Files: {} \n".format(infinite_loop_files))
            report.writelines(
                "=================================================================================================\n"
            )
            report.writelines("Failed Test & Infinite Loop Files: {} \n".format(test_failed_and_loop_files))
            report.writelines(
                "=================================================================================================\n"
            )

        df = pd.DataFrame(
            columns=["Source Language", "Target Language", "Filename", "BugType", "RootCause", "Impact", "Comments"]
        )
        index = 0
        for i in range(0, len(compile_failed_files)):
            list_row = [
                self.args.source_lang,
                self.args.target_lang,
                compile_failed_files[i],
                "",
                "",
                "Compilation Error",
                "",
            ]
            df.loc[i] = list_row
            index += 1
        for i in range(0, len(runtime_failed_files)):
            list_row = [
                self.args.source_lang,
                self.args.target_lang,
                runtime_failed_files[i],
                "",
                "",
                "Runtime Error",
                "",
            ]
            df.loc[index] = list_row
            index += 1
        for i in range(0, len(test_failed_files)):
            list_row = [self.args.source_lang, self.args.target_lang, test_failed_files[i], "", "", "Test Failed", ""]
            df.loc[index] = list_row
            index += 1
        for i in range(0, len(infinite_loop_files)):
            list_row = [
                self.args.source_lang,
                self.args.target_lang,
                infinite_loop_files[i],
                "",
                "",
                "Infinite Loop",
                "",
            ]
            df.loc[index] = list_row
            index += 1

        excel_fp = Path(self.report_dir).joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.dataset}_compileReport_from_"
            + str(self.args.source_lang)
            + "_to_"
            + str(self.args.target_lang)
            + ".xlsx",
        )
        df.to_excel(excel_fp, sheet_name="Sheet1")

        ordered_unsuccessful_fp = Path(self.report_dir).joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.dataset}_compileReport_from_"
            + str(self.args.source_lang)
            + "_to_"
            + str(self.args.target_lang)
            + f"_ordered_unsuccessful.txt",
        )

        # Statistics report

        run_info = {
            "Model": self.args.model,
            "Dataset": self.args.dataset,
            "Template": self.args.template_type,
            "Source PL": self.args.source_lang,
            "Target PL": self.args.target_lang,
            "Attempt": self.args.attempt,
        }
        run_info.update(statistics_dict)

        statistics_fp = Path(self.report_dir).joinpath(
            self.args.model, self.args.template_type, "execution_statistics.xlsx"
        )
        if os.path.isfile(statistics_fp):
            statistics_df = pd.read_excel(statistics_fp, index_col=0)
            statistics_df = pd.concat([statistics_df, pd.DataFrame([run_info])], ignore_index=True)
        else:
            statistics_df = pd.DataFrame(run_info, index=[0])
        statistics_df.to_excel(statistics_fp)

        self.ordered_unsuccessful_files = (
            compile_failed_files + runtime_failed_files + test_failed_files + infinite_loop_files
        )
        # setup ordered_unsuccessful:
        with open(ordered_unsuccessful_fp, "w") as f:
            for unsuccessful_instance in self.ordered_unsuccessful_files:
                f.write(f"{unsuccessful_instance}\n")

        for i, file in enumerate(self.infinite_loop):
            self.infinite_loop[i] = (
                file,
                "Error: The program did not finish execution before the timeout. An infinite loop was detected.",
            )

        attempt = self.args.attempt
        json_fp = Path(self.args.report_dir).joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.args.dataset}_errors_from_{self.args.source_lang}_to_{self.args.target_lang}_{attempt}.json",
        )
        with open(json_fp, "w", encoding="utf-8") as report:
            error_files = {
                "compile": self.compile_failed,
                "runtime": self.runtime_failed + self.infinite_loop,
                "incorrect": self.test_failed,
                "token_exceeded": self.token_exceeded,
            }
            json.dump(error_files, report)
            report.close()

        txt_fp = Path(self.report_dir).joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.args.dataset}_errors_from_{self.args.source_lang}_to_{self.args.target_lang}_{attempt}.txt",
        )
        with open(txt_fp, "w") as report:
            for i in range(len(self.ordered_unsuccessful_files)):
                if self.ordered_unsuccessful_files[i] in get_filenames_from(self.compile_failed):
                    print("0,Compilation Error", file=report)
                elif self.ordered_unsuccessful_files[i] in get_filenames_from(self.runtime_failed):
                    print("0,Runtime Error", file=report)
                elif self.ordered_unsuccessful_files[i] in get_filenames_from(self.test_failed):
                    print("0,Wrong Output", file=report)
                elif self.ordered_unsuccessful_files[i] in self.test_passed:
                    print("1,Fixed", file=report)
                elif self.ordered_unsuccessful_files[i] in self.token_exceeded:
                    print("0,Token Exceeded", file=report)
                elif self.ordered_unsuccessful_files[i] in get_filenames_from(self.infinite_loop):
                    print("0,Infinite Loop", file=report)
                else:
                    print("0,Unknown", file=report)


def get_filenames_from(list_of_tuples: list[tuple | list]):
    return [x[0] for x in list_of_tuples]


class ExtendedCompileAndTest(CompileAndTest):
    def __init__(self, args: CLIArgumentsExecution, config: Config):
        super().__init__(args, config)
