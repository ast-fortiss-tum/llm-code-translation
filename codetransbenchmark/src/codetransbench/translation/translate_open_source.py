from io import TextIOWrapper
import os
import logging
import traceback
from dotenv import load_dotenv
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import codetrans


from codetransbench.utils.cli_abstraction import CLIArgumentsTranslation

import codetrans.code_encoding_sanitization
from codetransbench.utils.config import Config, load_config
from codetransbench.utils.metadata import TranslationMetadata

logger = logging.getLogger(__name__)

FILE_EXTENSIONS = {
    "Python": "py",
    "C": "c",
    "C++": "cpp",
    "Java": "java",
    "Go": "go",
    "Rust": "rs",
    "C#": "cs",
}


class CodeTranslator:
    def __init__(self, args: CLIArgumentsTranslation, config: Config):
        self.args = args
        self.config = config
        self.set_model_name_engine()

    def set_model_name_engine(self):
        if "llamafile" in self.args.model:
            self.model_name = self.args.model.removeprefix("llamafile_")
            print("Name for the llamafile model:", self.model_name)
            self.model_engine = "llamafile"
        elif "ollama" in self.args.model:
            self.model_name = self.args.model.removeprefix("ollama_")
            print("Name for the ollama model:", self.model_name)
            self.model_engine = "ollama"
        elif "langchain" in self.args.model:
            self.model_name = self.args.model.removeprefix("langchain_")
            print("Name for the langchain model:", self.model_name)
            self.model_engine = "torch"
        else:
            raise NotImplementedError("The given model was not implemented.")
        return self.model_name, self.model_engine

    def setup_files(self):
        self.input_dir = self.config.dataset_dir / self.args.dataset / self.args.source_lang / "Code"
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory {str(self.input_dir)} does not exist.")

        self.main_output_path = (
            self.config.output_dir / f"{self.args.model}_{self.args.template_type}" / "iteration_1" / self.args.dataset
        )
        self.out_folder = self.main_output_path / self.args.source_lang / self.args.target_lang
        os.makedirs(self.out_folder, exist_ok=True)

        self.in_files = os.listdir(self.input_dir)
        print(f"found {len(self.in_files)} inputs")

        self.find_already_translated_files()
        self.find_erroneous_source_files()

        if len(self.already_extracted_files) > 0 or len(self.erroneous_source_files) > 0:
            self.in_files = [
                x
                for x in self.in_files
                if (x not in self.erroneous_source_files) and (x.split(".")[0] not in self.already_extracted_files)
            ]

    def find_already_translated_files(self):
        self.already_extracted_files = []
        if os.path.exists(self.out_folder):
            self.already_extracted_files = os.listdir(self.out_folder)
            if len(self.already_extracted_files) > 0:
                self.already_extracted_files = [
                    x.split(".")[0] for x in self.already_extracted_files if os.stat(self.out_folder / x).st_size != 0
                ]

    def find_erroneous_source_files(self):
        erroneous_files_path = (
            self.config.testresults_dir
            / "llamafile_x"
            / "test_source"
            / "iteration_1"
            / f"llamafile_x_{self.args.dataset}_compileReport_from_{self.args.source_lang}_to_{self.args.source_lang}_ordered_unsuccessful.txt"
        )
        self.erroneous_source_files = []
        if os.path.exists(erroneous_files_path):
            with open(erroneous_files_path, "r") as f:
                self.erroneous_source_files = [x.strip() for x in f.readlines()]

    def setup_llm(self):
        llm_settings = codetrans.llm_abstraction.LLMSettings(
            top_k=self.args.k,
            top_p=self.args.p,
            temperature=self.args.temperature,
            repeat_penalty=1,
        )
        self.llm = codetrans.llm_wrapper(self.model_name, self.model_engine, llm_settings=llm_settings)
        self.save_model_metadata(llm_settings)

    def save_model_metadata(self, llm_settings):
        tm = TranslationMetadata(self.model_name, self.model_engine, llm_settings, [self.args.template_type])
        tm.save_to_file(self.main_output_path / "metadata.yml")

    def run(self):
        # loop over input files
        sanitisation_report_path = f"sanitisation_report_{self.args.model}.txt"
        san_rep_file = open(sanitisation_report_path, "a")
        context_window_report = f"context_window_report_{self.args.model}.txt"
        context_window_file = open(context_window_report, "a")
        sanitised_counter = 0
        for f in tqdm(self.in_files):
            source_code_file = self.input_dir / f

            source_code_str = ""
            with open(source_code_file, "r", encoding="UTF-8", errors="ignore") as fin:
                source_code_str = fin.read()

            # sanitise source code characters
            sanitised = self.sanitise_characters_of_code(f, source_code_str, sanitised_counter, san_rep_file)
            if not sanitised:
                # There was an UnicodeError, the file will be skipped
                print("Skip")
                continue
            sanitised_counter, source_code_str = sanitised

            try:
                t0 = time.perf_counter()

                raw_outputs = self.translate_source_code(source_code_str)

                t1 = time.perf_counter()

                if "# Token size exceeded" in raw_outputs:
                    context_window_file.write(f"{raw_outputs} for file {f}\n")

                out_file = self.out_folder / f'{f.split(".")[0]}.{FILE_EXTENSIONS[self.args.target_lang]}'
                print(f"\n{time.ctime()}: {out_file} Total generation time:", t1 - t0)
                with open(out_file, "w") as fot:
                    print(raw_outputs, file=fot)

            except (ValueError, FileNotFoundError) as e:
                print(e)
                continue
        san_rep_file.write(f"Total sanitised {sanitised_counter}\n for {self.args.source_lang} source code.\n")

    def translate_source_code(self, source_code_str):
        templates = codetrans.llm_chain.create_prompt_template_for_model(self.args.template_type, self.model_name)

        prompt = codetrans.llm_chain.fillin_prompt_template(
            templates[0],
            source_code_str,
            self.args.source_lang,
            self.args.target_lang,
        )

        max_output_tokens = codetrans.llm_chain.check_context_size(prompt, self.model_name)
        if max_output_tokens <= 0:
            logger.info(f"The tokens exceeded the maximum size of the context window by {max_output_tokens} tokens.")
            return f"# Token size exceeded by {-max_output_tokens} tokens"

        if self.args.template_type == "via_description":
            result = codetrans.llm_chain.create_and_invoke_via_description_chain(
                templates,
                self.llm,
                source_code_str,
                self.args.source_lang,
                self.args.target_lang,
            )
        elif self.args.template_type == "via_description_1_shot":
            result = codetrans.llm_chain.create_and_invoke_via_description_w_example_chain(
                templates,
                self.llm,
                source_code_str,
                self.args.source_lang,
                self.args.target_lang,
            )
        else:
            result = codetrans.llm_chain.create_and_invoke_llm_chain(
                templates[0],
                self.llm,
                source_code_str,
                self.args.source_lang,
                self.args.target_lang,
            )
        raw_outputs = result["target_code"]
        return raw_outputs

    def sanitise_characters_of_code(
        self,
        filename: str,
        source_code: str,
        sanitised_counter: int = 0,
        sanitisation_report: TextIOWrapper | None = None,
    ) -> tuple[int, str] | None:
        file_path = self.input_dir / filename

        try:
            # prompt_str_sanitised = codetrans.remove_non_ISO_8859_1_characters_in_comments(prompt_str, args.source_lang)
            source_code_sanitised = codetrans.code_encoding_sanitization.remove_non_ASCII_characters_in_comments(
                source_code, self.args.source_lang
            )
        except Exception as ex:
            if "comment_parser.comment_parser.ParseError" in traceback.format_exc():
                max_repeats = 15
                source_code_sanitised = codetrans.postprocessing.repeated_lines(source_code, max_repeats)
                if source_code_sanitised != source_code:
                    logger.warning(f"Cut of file {filename} after {max_repeats} identical lines in a row.")
                    try:
                        source_code_sanitised = (
                            codetrans.code_encoding_sanitization.remove_non_ASCII_characters_in_comments(
                                source_code_sanitised, self.args.source_lang
                            )
                        )
                    except Exception as ex:
                        logger.exception(ex)
                        logger.warning(
                            f"Skipped sanitisation for file {filename} because of some error with the comment extraction."
                        )
                else:
                    logger.exception(ex)
                    logger.warning(
                        f"Skipped sanitisation for file {filename} because of some error with the comment extraction."
                    )
            else:
                logger.exception(ex)
                source_code_sanitised = source_code
                logger.warning(
                    f"Skipped sanitisation for file {filename} because of some error with the comment extraction."
                )

        if source_code_sanitised != source_code:
            print(f"\nThe file {filename} was sanitised for {self.args.source_lang} to {self.args.target_lang}.")
            sanitised_counter += 1

            if sanitisation_report is not None:
                sanitisation_report.write(f"sanitised {filename}\n")

            # use the sanitised version of the prompt
            source_code = source_code_sanitised

        # use the initial encoding used for the requests
        try:
            source_code = source_code.encode("ISO-8859-1").decode("ISO-8859-1")
        except UnicodeEncodeError:
            with open("unicode_exceptions.txt", "a+") as unic_exc_file:
                print(
                    f"{time.ctime()}: <FILE>{file_path}</FILE>: For {self.args.source_lang} code to {self.args.target_lang} for model {self.args.model}",
                    file=unic_exc_file,
                )
            print(f"\n{time.ctime()}: {file_path} skipped because of UnicodeEncodeError")

            return None

        return sanitised_counter, source_code


def main(args: CLIArgumentsTranslation, config: Config):

    translator = CodeTranslator(args, config)

    translator.setup_files()

    translator.setup_llm()

    translator.run()


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="run translation with open-source models given dataset and languages")
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
        help="type of the prompt template to use for code translation.",
        required=True,
        type=str,
    )
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

    # nsp = CLIArgumentsTranslation()
    # args = parser.parse_args(namespace=nsp)

    args = CLIArgumentsTranslation(**vars(parser.parse_args()))

    config = load_config()
    main(args, config)
