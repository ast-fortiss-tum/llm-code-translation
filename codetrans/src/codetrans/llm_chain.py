from codetrans.codetrans_config import TORCH_MODELS_PATH
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain_core.language_models.llms import LLM
from pathlib import Path
from transformers import AutoTokenizer

from codetrans.llm_abstraction import LLAMAFILE_CTX_SIZE
from codetrans.postprocessing import comment_syntax, markdown_pl_mapping_bidict
from codetrans.prompt_templates import (
    direct_translation_template,
    via_description_translation_template_1,
    via_description_translation_template_2,
    via_description_translation_template_1_w_example,
    translation_allowing_questions_template,
    vanilla_template,
    lit_paper_naive_template,
    controlled_template,
    controlled_md_template,
    direct_md_template,
    controlled_md_no_system_template,
    short_md_template,
    example_description,
    example_source_code,
    compile_runtime_error_evalplus_template,
    compile_runtime_error_template,
    test_failure_evalplus_template,
    test_failure_io_based_template,
)

# Template switch

template_type = "direct"


def hf_modelfiles_path_for(model_name: str) -> Path:
    model_name = model_name.lower()
    hf_model_paths = {
        "mistral": Path.joinpath(TORCH_MODELS_PATH, "Mistral-7B-Instruct-v0.1"),
        "mixtral": Path.joinpath(TORCH_MODELS_PATH, "Mixtral-8x7B-Instruct-v0.1"),
        "codellama": Path.joinpath(TORCH_MODELS_PATH, "CodeLlama-70b-hf"),
        "dolphin-2.6-mistral": Path.joinpath(TORCH_MODELS_PATH, "dolphin-2.6-mistral-7b"),
        "dolphin-2.7-mixtral": Path.joinpath(TORCH_MODELS_PATH, "dolphin-2.7-mixtral-8x7b"),
        "dolphincoder-starcoder2-15b": Path.joinpath(TORCH_MODELS_PATH, "dolphincoder-starcoder2-15b"),
        "dolphin-2.6-phi-2": Path.joinpath(TORCH_MODELS_PATH, "dolphin-2_6-phi-2"),
        "llama3": Path.joinpath(TORCH_MODELS_PATH, "Meta-Llama-3-8B-Instruct"),
        "phi3": Path.joinpath(TORCH_MODELS_PATH, "Phi-3-mini-4k-instruct"),
        "codestral": Path.joinpath(TORCH_MODELS_PATH, "Codestral-22B-v0.1"),
    }

    if model_name not in hf_model_paths.keys():
        raise NotImplementedError(
            f"The model you are trying to use is not available in this library. Model: {model_name}"
        )

    return hf_model_paths[model_name]


def apply_chat_template_to_text(text: str, model_name: str) -> str:
    if "codestral" in model_name:
        # The codestral tokenizer does not define a chat template. Codestral uses the same chat template as Mistral. Use that instead.
        tokenizer = AutoTokenizer.from_pretrained(hf_modelfiles_path_for("mistral"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_modelfiles_path_for(model_name))
    if "dolphin" in model_name:
        # has no chat template in tokenizer
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        system_prompt = "You are a skilled software developer proficient in multiple programming languages."
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text.removeprefix(system_prompt + " ")},
        ]

    elif "llama3" in model_name:
        return text
    else:
        chat = [
            {"role": "user", "content": text},
        ]
    return tokenizer.apply_chat_template(chat, tokenize=False)


def check_context_size(text: str, model_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(hf_modelfiles_path_for(model_name))
    tokens = tokenizer.encode(text)
    # print(tokens)

    total_input_tokens = len(tokens)
    print("Total input tokens:", total_input_tokens)
    model_max_length = LLAMAFILE_CTX_SIZE[model_name]
    if total_input_tokens >= model_max_length:
        return model_max_length - total_input_tokens
    max_new_tokens = model_max_length - total_input_tokens
    return max_new_tokens


def apply_chat_template_to_prompt_template(template: PromptTemplate, model_name: str) -> PromptTemplate:
    """Apply a LLMs chat template to the text of a prompt template.

    Args:
        template (PromptTemplate): A `PromptTemplate` object containing the text of the prompt template.
        model_name (str): The name of the LLM that will be applied to the prompt template.

    Returns:
        PromptTemplate: A new `PromptTemplate` object with the updated text resulting from applying the chat template to the original prompt template.
    """
    template.template = apply_chat_template_to_text(template.template, model_name)
    return template


def create_prompt_template_for_model(template_type: str, model_name: str) -> list[PromptTemplate]:
    """
    Create a prompt template for a given template type and model name.

    Args:
        template_type (str): The type of prompt template to create.
        model_name (str): The name of the model for with to apply its chat template.

    Returns:
        list[PromptTemplate]: A list of modified prompt templates that have been embedded with the given chat template.
    """
    prompts = create_prompt_template(template_type)
    modified_prompts = [apply_chat_template_to_prompt_template(p, model_name) for p in prompts]
    return modified_prompts


def create_prompt_template(template_type: str) -> list[PromptTemplate]:
    """Creates a list of PromptTemplate objects based on the given template type.

    Args:
        template_type (str): The template type to use for creating prompts.

    Returns:
        list[PromptTemplate]: A list of PromptTemplate objects with the corresponding input variables.
    """
    prompts = []

    match template_type:
        case "direct":
            template = direct_translation_template
            input_var = ["source_code", "source_pl", "target_pl"]
        case "via_description":
            template = via_description_translation_template_1
            input_var = ["source_code", "source_pl"]
            template_2 = via_description_translation_template_2
            input_var_2 = ["description", "target_pl"]
            prompts.append(PromptTemplate(template=template_2, input_variables=input_var_2))
        case "via_description_1_shot":
            template = via_description_translation_template_1_w_example
            input_var = ["source_code", "source_pl", "example_description", "example_source_code"]
            template_2 = via_description_translation_template_2
            input_var_2 = ["description", "target_pl"]
            prompts.append(PromptTemplate(template=template_2, input_variables=input_var_2))
        case "allowing_questions":
            template = translation_allowing_questions_template
            input_var = ["source_code", "source_pl", "target_pl"]
        case "lit":
            template = lit_paper_naive_template
            input_var = ["source_code", "source_pl", "target_pl"]
        case "vanilla":
            template = vanilla_template
            input_var = ["source_code", "source_pl", "target_pl"]
        case "controlled":
            template = controlled_template
            input_var = ["source_code", "source_pl", "target_pl"]
        case "controlled_md":
            template = controlled_md_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "target_pl_comment"]
        case "direct_md":
            template = direct_md_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "target_pl_comment"]
        case "controlled_md_no_system":
            template = controlled_md_no_system_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "target_pl_comment"]
        case "short_md":
            template = short_md_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "target_pl_comment"]
        case "compile_runtime_error_evalplus":
            template = compile_runtime_error_evalplus_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "translated_code", "stderr"]
        case "compile_runtime_error":
            template = compile_runtime_error_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "translated_code", "stderr"]
        case "test_failure_evalplus":
            template = test_failure_evalplus_template
            input_var = ["source_code", "source_pl", "target_pl", "target_pl_md", "translated_code", "stderr"]
        case "test_failure_io_based":
            template = test_failure_io_based_template
            input_var = [
                "source_code",
                "source_pl",
                "target_pl",
                "target_pl_md",
                "translated_code",
                "generated_output",
                "test_inputs",
                "test_outputs",
            ]
        case _:
            raise ValueError(f"The given template type does not exist: {template_type}")

    prompts.insert(0, PromptTemplate(template=template, input_variables=input_var))
    return prompts


def fillin_prompt_template(
    prompt: PromptTemplate,
    source_code: str,
    source_pl: str,
    target_pl: str,
    translated_code: str = "",
    stderr: str = "",
    test_data: dict[str, str] = {},
) -> dict[str, str]:
    """
    Fills the prompt template with the given input values.

    Args:
        prompt (PromptTemplate): A prompt template that will be used to generate the LLM chain.
        llm (LLM): An LLM model that will be used in the LLM chain.
        source_code (str): The source code for which a translation is needed.
        source_pl (str): The source language for the source code.
        target_pl (str): The target language for the translation.
        tranlated_code (str): The current translation of the source code in the target language.
        stderr (str): The error information of standard error of the latest execution.
        test_data (dict): The data from the latest test execution (input, expected output, and generated output).

    Returns:
        The filled in prompt template of the chain as a string.
    """
    return prompt.format(
        source_code=source_code,
        source_pl=source_pl,
        target_pl=target_pl,
        target_pl_md=markdown_pl_mapping_bidict()[target_pl],
        target_pl_comment=comment_syntax(target_pl)[0],
        translated_code=translated_code,
        stderr=stderr,
        generated_output=test_data.get("generated_output", ""),
        test_inputs=test_data.get("test_inputs", ""),
        test_outputs=test_data.get("test_outputs", ""),
    )


def create_and_invoke_llm_chain(
    prompt: PromptTemplate,
    llm: LLM,
    source_code: str,
    source_pl: str,
    target_pl: str,
    translated_code: str = "",
    stderr: str = "",
    test_data: dict[str, str] = {},
) -> dict[str, str]:
    """
    Creates an LLMChain using a given prompt template and LLM object.

    Args:
        prompt (PromptTemplate): A prompt template that will be used to generate the LLM chain.
        llm (LLM): An LLM model that will be used in the LLM chain.
        source_code (str): The source code for which a translation is needed.
        source_pl (str): The source language for the source code.
        target_pl (str): The target language for the translation.
        tranlated_code (str): The current translation of the source code in the target language.
        stderr (str): The error information of standard error of the latest execution.
        test_data (dict): The data from the latest test execution (input, expected output, and generated output).

    Returns:
        A dictionary containing the translated source code and its corresponding source code in the specified languages.
    """
    # create prompt template > LLM chain
    chain = LLMChain(prompt=prompt, llm=llm, output_key="target_code")  # the same as:  prompt | llm
    # Invoke the chain
    return chain.invoke(
        {
            "source_code": source_code,
            "source_pl": source_pl,
            "target_pl": target_pl,
            "target_pl_md": markdown_pl_mapping_bidict()[target_pl],
            "target_pl_comment": comment_syntax(target_pl)[0],
            "translated_code": translated_code,
            "stderr": stderr,
            "generated_output": test_data.get("generated_output", ""),
            "test_inputs": test_data.get("test_inputs", ""),
            "test_outputs": test_data.get("test_outputs", ""),
        }
    )


def create_and_invoke_via_description_chain(
    prompts: list[PromptTemplate],
    llm: LLM,
    source_code: str,
    source_pl: str,
    target_pl: str,
) -> dict[str, str]:
    """
    Create and invoke a chain for code translation via a description.

    Args:
        llm (LLM): The LLM model to use.
        source_code (str): The source code to be translated.
        source_pl (str): The source language of the code to be translated.
        target_pl (str): The target language of the code to be translated.

    Returns:
        A dictionary containing the description and translated code in the target programming language.
    """
    # first step in chain
    chain_one = LLMChain(llm=llm, prompt=prompts[0], output_key="description")

    # second step in chain
    chain_two = LLMChain(llm=llm, prompt=prompts[1], output_key="target_code")

    # Combine the first and the second chain
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["source_code", "source_pl", "target_pl"],
        output_variables=["description", "target_code"],
        verbose=True,
    )

    translation_result = overall_chain.invoke(
        {"source_code": source_code, "source_pl": source_pl, "target_pl": target_pl}
    )
    return translation_result


def create_and_invoke_via_description_w_example_chain(
    prompts: list[PromptTemplate],
    llm: LLM,
    source_code: str,
    source_pl: str,
    target_pl: str,
) -> dict[str, str]:
    """
    Create and invoke a chain for code translation via a description.

    Args:
        llm (LLM): The LLM model to use.
        source_code (str): The source code to be translated.
        source_pl (str): The source language of the code to be translated.
        target_pl (str): The target language of the code to be translated.

    Returns:
        A dictionary containing the description and translated code in the target programming language.
    """
    # first step in chain
    chain_one = LLMChain(llm=llm, prompt=prompts[0], output_key="description")

    # second step in chain
    chain_two = LLMChain(llm=llm, prompt=prompts[1], output_key="target_code")

    # Combine the first and the second chain
    overall_chain = SequentialChain(
        chains=[chain_one, chain_two],
        input_variables=["source_code", "source_pl", "target_pl", "example_description", "example_source_code"],
        output_variables=["description", "target_code"],
        verbose=True,
    )

    translation_result = overall_chain.invoke(
        {
            "source_code": source_code,
            "source_pl": source_pl,
            "target_pl": target_pl,
            "example_description": example_description,
            "example_source_code": example_source_code,
        }
    )
    return translation_result


# MAYBE: use different prompt structure per LLM???
