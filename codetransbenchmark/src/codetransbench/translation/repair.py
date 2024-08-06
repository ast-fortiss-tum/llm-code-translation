import os
import time
import logging

import json
from dotenv import load_dotenv
import torch
from tqdm import tqdm
import argparse

import codetrans

from codetransbench.translation.translate_open_source import CodeTranslator, FILE_EXTENSIONS
from codetransbench.utils.cli_abstraction import CLIArgumentsTranslationIteration
from codetransbench.utils.config import Config, load_config

logger = logging.getLogger(__name__)


class CodeTranslationRepair(CodeTranslator):

    def __init__(self, args: CLIArgumentsTranslationIteration, config: Config) -> None:
        super().__init__(args, config)
        self.args = args

    def setup_files(self):
        self.main_dir = self.config.base_dir
        # the directory where the cleaned translations are stored
        self.translation_dir = (
            self.config.output_dir
            / self.config.cleaned_dir_name
            / self.args.model
            / self.args.template_type
            / f"iteration_{self.args.attempt}"
            / self.args.dataset
            / self.args.source_lang
            / self.args.target_lang
        )

        # input directory with all the code examples to translate
        self.input_dir = self.config.dataset_dir / self.args.dataset / self.args.source_lang / "Code"
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory {str(self.input_dir)} does not exist.")

        # Make a directory at output/model_template/iteration_attempt/<dataset>
        self.main_output_path = (
            self.config.output_dir
            / f"{self.args.model}_{self.args.template_type}"
            / f"iteration_{self.args.attempt + 1}"
            / self.args.dataset
        )
        self.out_folder = self.main_output_path / self.args.source_lang / self.args.target_lang
        os.makedirs(self.out_folder, exist_ok=True)

    def setup_errors(self):

        # set errors
        self.errors = {}
        json_fp = self.config.testresults_dir.joinpath(
            self.args.model,
            self.args.template_type,
            f"iteration_{self.args.attempt}",
            f"{self.args.model}_{self.args.dataset}_errors_from_{self.args.source_lang}_to_{self.args.target_lang}_{self.args.attempt}.json",
        )
        with open(json_fp, "r") as f:
            self.errors: dict = json.load(f)

    def translate_source_code(
        self,
        source_code,
        translated_code,
        stderr,
        test_inputs,
        test_outputs,
        generated,
        error_type,
    ) -> str:
        if error_type in ["compile", "runtime"] and self.args.dataset == "evalplus":
            template_type = "compile_runtime_error_evalplus"
        elif error_type in ["compile", "runtime"]:
            template_type = "compile_runtime_error"
        elif error_type == "incorrect" and self.args.dataset == "evalplus":
            template_type = "test_failure_evalplus"
        elif error_type == "incorrect":
            template_type = "test_failure_io_based"
        elif error_type == "token_exceeded":
            return "# Token size exceeded."
        else:
            ValueError("There is no template to fix this error combination.")
        templates = codetrans.llm_chain.create_prompt_template_for_model(template_type, self.model_name)

        prompt = codetrans.llm_chain.fillin_prompt_template(
            templates[0],
            source_code,
            self.args.source_lang,
            self.args.target_lang,
            translated_code,
            stderr,
            {"test_inputs": test_inputs, "test_outputs": test_outputs, "generated_output": generated},
        )

        max_output_tokens = codetrans.llm_chain.check_context_size(prompt, self.model_name)
        if max_output_tokens <= 0:
            logger.info(f"The tokens exceeded the maximum size of the context window by {max_output_tokens} tokens.")
            return "# Token size exceeded."

        result = codetrans.llm_chain.create_and_invoke_llm_chain(
            templates[0],
            self.llm,
            source_code,
            self.args.source_lang,
            self.args.target_lang,
            translated_code,
            stderr,
            {"test_inputs": test_inputs, "test_outputs": test_outputs, "generated_output": generated},
        )
        return result["target_code"]

    def run(self):

        for error_type in self.errors.keys():
            logger.info(f"Fixing {error_type} errors.")

            snippets = self.errors[error_type]
            for snippet in tqdm(snippets, total=len(snippets)):

                if isinstance(snippet, str):
                    snippet = [
                        snippet,
                        "Error: The program did not finish execution before the timeout. An infinite loop was detected.",
                    ]

                for i, value in enumerate(snippet):
                    snippet[i] = value.strip()

                stderr, test_inputs, test_outputs, generated_output = "", "", "", ""

                if error_type in ["compile", "runtime"]:
                    filename, stderr = snippet

                elif error_type == "incorrect" and self.args.dataset == "evalplus":
                    filename, stderr = snippet

                elif error_type == "incorrect":
                    filename, test_inputs, test_outputs, generated_output = snippet

                else:
                    raise NotImplementedError(
                        f"The given error_type {error_type} is not implemented. Valid values: [compile,runtime,incorrect]"
                    )

                # do not fix the file again in the same iteration
                fixed_code_file = self.out_folder / filename
                if fixed_code_file.exists() and self.check_errors_for_file(filename, snippet, error_type):
                    logger.info(f"{filename} already correct.")
                    print(f"{filename} already correct.")
                    continue

                filename_without_extension = filename.split(".")[0]

                # input source code in the source language

                source_code_file = (
                    self.input_dir / f"{filename_without_extension}.{FILE_EXTENSIONS[self.args.source_lang]}"
                )
                source_code = ""
                with open(source_code_file, "r", encoding="UTF-8", errors="ignore") as f:
                    source_code = f.read().strip()

                # sanitise source code characters
                sanitised = self.sanitise_characters_of_code(source_code_file, source_code)
                if not sanitised:
                    # There was an UnicodeError, the file will be skipped
                    print("Skip")
                    continue
                _, source_code = sanitised

                recent_translated_code = ""
                with open(self.translation_dir / filename, "r", encoding="UTF-8", errors="ignore") as f:
                    recent_translated_code = f.read().strip()

                sanitised = self.sanitise_characters_of_code(self.translation_dir / filename, recent_translated_code)
                if not sanitised:
                    # There was an UnicodeError, the file will be skipped
                    raise UnicodeError(
                        "This unicode error should not exist. The LLM translation contained illegal characters"
                    )
                    continue
                _, recent_translated_code = sanitised

                # translate the source code again with the error messages provided

                try:
                    t0 = time.perf_counter()

                    translated_code = self.translate_source_code(
                        source_code,
                        recent_translated_code,
                        stderr,
                        test_inputs,
                        test_outputs,
                        generated_output,
                        error_type,
                    )

                    t1 = time.perf_counter()

                    print(f"\n{time.ctime()}: {fixed_code_file} Total generation time:", t1 - t0)
                    with open(fixed_code_file, "w") as f:
                        print(translated_code, file=f)

                except (ValueError, FileNotFoundError) as e:
                    print(e)
                    continue

    def check_errors_for_file(self, current_filename: str, current_snippet: list[str] | tuple[str], error_type: str):
        """Helper to check if the error message is the same for a file with the tests run on the updated testbed.
        If the error changed the repair step has to be run again.
        """
        self.old_errors = {}
        json_fp = self.config.testresults_dir.joinpath(
            self.args.model,
            self.args.template_type,
            f"old_iteration_{self.args.attempt}",
            f"{self.args.model}_{self.args.dataset}_errors_from_{self.args.source_lang}_to_{self.args.target_lang}_{self.args.attempt}.json",
        )
        if os.path.exists(json_fp):
            with open(json_fp, "r") as f:
                self.old_errors: dict = json.load(f)
        else:
            # the repair was not done before
            return True

        if self.old_errors:
            logger.info(f"Looking up error information for {current_filename} ...")

            snippets = self.old_errors[error_type]
            for snippet in snippets:

                if isinstance(snippet, str):
                    snippet = [
                        snippet,
                        "Error: The program did not finish execution before the timeout. An infinite loop was detected.",
                    ]

                for i, value in enumerate(snippet):
                    snippet[i] = value.strip()

                if (
                    (current_filename == snippet[0])
                    and (len(current_snippet) == len(snippet))
                    and all(x == y and type(x) == type(y) for x, y in zip(current_snippet, snippet))
                ):
                    return True
        return False


def main(args: CLIArgumentsTranslationIteration, config: Config):

    translator = CodeTranslationRepair(args, config)

    translator.setup_files()

    translator.setup_llm()

    translator.setup_errors()

    translator.run()


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="run repair with a given model, dataset and languages")
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
        help="type of the prompt template used for code translation.",
        required=True,
        type=str,
    )
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
    parser.add_argument(
        "--k",
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering. Only applies for sampling mode, with range from 1 to 100.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--p",
        help="Only the most probable tokens with probabilities that add up to top_p or higher are considered during decoding. The valid range is 0.0 to 1.0. 1.0 is equivalent to disabled and is the default. Only applies to sampling mode. Also known as nucleus sampling.",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--temperature",
        help='A value used to warp next-token probabilities in sampling mode. Values less than 1.0 sharpen the probability distribution, resulting in "less random" output. Values greater than 1.0 flatten the probability distribution, resulting in "more random" output. A value of 1.0 has no effect and is the default. The allowed range is 0.0 to 2.0.',
        required=True,
        type=float,
    )
    parser.add_argument("--attempt", help="Attempt number to repair.", required=True, type=int)
    # nsp = CLIArgumentsTranslationIteration()

    # args = parser.parse_args(namespace=nsp)
    args = CLIArgumentsTranslationIteration(**vars(parser.parse_args()))

    config = load_config()
    main(args, config)
