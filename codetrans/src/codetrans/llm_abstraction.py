from dataclasses import dataclass
from pathlib import Path
import platform
import subprocess
import time
from codetrans import utils

from codetrans.codetrans_config import (
    GGUF_PATH,
    LLAMAFILE_OUTPUT_LOG,
    LLAMAFILE_PATH,
    LLAMAFILE_VERSION,
    TORCH_MODELS_PATH,
)
from langchain_core.language_models.llms import LLM
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.llamafile import Llamafile
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from io import TextIOWrapper
from pathlib import Path
import requests


MAX_CTX_SIZE = 100000
"""Max size of the prompt context."""
LLAMAFILE_PORTS = {
    "mistral": 8090,
    "mixtral": 8091,
    "codellama": 8092,
    "dolphin-2.6-mistral": 8093,
    "dolphin-2.7-mixtral": 8094,
    "dolphincoder-starcoder2-15b": 8095,
    "dolphin-2.6-phi-2": 8096,
    "llama3": 8097,
    "phi3": 8098,
    "codestral": 8099,
}
"""Mapping of the LLM names and the port numbers the respective llamafile servers run on."""
LLAMAFILE_CTX_SIZE = {
    "mistral": 8192,
    "mixtral": 8192,
    "codellama": 100000,
    "dolphin-2.6-mistral": 16000,
    "dolphin-2.7-mixtral": 16000,
    "dolphincoder-starcoder2-15b": 4000,
    "dolphin-2.6-phi-2": 2048,
    "llama3": 8000,
    "phi3": 4000,
    "codestral": 32000,
}

OLLAMA_CTX_SIZE = {
    "mistral": 8192,
    "mixtral:8x7b": 8192,
    "codellama:70b": 100000,
    "dolphin-mistral": 16000,
    "dolphin-mixtral": 16000,
    "llama3": 8000,
    "phi3": 4000,
    "codestral": 32000,
}


llm_abstraction_mode = "llamafile"  # "ollama" # "torch"


@dataclass
class LLMSettings:
    top_k: int
    top_p: float
    temperature: float
    repeat_penalty: float


def on_windows_workstation():
    return platform.system() == "Windows"


def on_macbook():
    return platform.system() == "Darwin"


def hardware_for_os(hardware_mode: str):
    if on_macbook():
        # always use the GPU on the MacBook
        return "mps", "mps:0"
    if hardware_mode.lower() == "gpu":
        return "cuda", 0
    elif hardware_mode.lower() == "cpu":
        return "cpu", "cpu"
    else:
        raise NotImplementedError(f"The requested hardware_mode {hardware_mode} is not implemented yet.")


valid_abstractions = ["ollama", "torch", "llamafile"]


def llm_wrapper(
    model_name: str,
    abstraction_framework: str = "llamafile",
    hardware_mode: str = "cpu",
    max_output_tokens: int = 1200,
    llm_settings: LLMSettings | None = None,
) -> Ollama | Llamafile | HuggingFacePipeline:

    if abstraction_framework == "ollama":
        # Raises ValidationError if the input data cannot be parsed to form a valid model.
        if llm_settings:
            llm = Ollama(
                model=model_name,
                num_ctx=OLLAMA_CTX_SIZE[model_name],
                num_predict=-2,
                top_k=llm_settings.top_k,
                top_p=llm_settings.top_p,
                temperature=llm_settings.temperature,
                repeat_penalty=llm_settings.repeat_penalty,
            )
        else:
            # num_predict = -2 --> fill the context window
            llm = Ollama(model=model_name, num_ctx=OLLAMA_CTX_SIZE[model_name], num_predict=-2)
        print("Serving model via Ollama")
    elif abstraction_framework == "llamafile":
        if not llamafile_server_for_model_exists(model_name):
            lf_process, outfile = start_llamafile(model_name, ctx_size=LLAMAFILE_CTX_SIZE[model_name.lower()])
            # print("after start_llamafile")
            time.sleep(5)
            # wait until the llamafile server is ready
            while (
                LLAMAFILE_VERSION == "0.6"
                and not llamafile_server_for_model_exists(model_name)
                and not simple_llamafile_server_ready(model_name)
            ):
                print("llamafile not ready")
                time.sleep(2)
            # wait until the server status is "running and ok"
            status = check_llamafile_status(lf_process, model_name)
            while LLAMAFILE_VERSION == "0.6.2" and status != "running and ok":
                print("llamafile status:", status)
                if "error" in status:
                    # kill and restart
                    print("Restarting llamafile")
                    kill_llamafile_server(lf_process, outfile, model_name)
                    time.sleep(1)
                    lf_process = start_llamafile(model_name, ctx_size=LLAMAFILE_CTX_SIZE[model_name.lower()])
                elif "loading model" in status:
                    time.sleep(1)
                elif "no slot available" in status:
                    time.sleep(1)
                else:
                    # what happens in the other cases?
                    # raise NotImplementedError(f"No idea yet: status: {status}")
                    time.sleep(1)
                    print("llamafile is starting. status:", status)
                status = check_llamafile_status(lf_process, model_name)
        # MAYBE provide parameters
        port = get_llamafile_port_for_model(model_name)
        if llm_settings:
            llm = Llamafile(
                base_url=f"http://localhost:{port}",
                n_predict=LLAMAFILE_CTX_SIZE[model_name.lower()],
                top_k=llm_settings.top_k,
                top_p=llm_settings.top_p,
                temperature=llm_settings.temperature,
                repeat_penalty=llm_settings.repeat_penalty,
            )
        else:
            llm = Llamafile(
                base_url=f"http://localhost:{port}",
                n_predict=LLAMAFILE_CTX_SIZE[model_name.lower()],
            )
        print("Serving model via Llamafile:", model_name)
    elif abstraction_framework == "torch":
        torch.set_default_device(hardware_for_os(hardware_mode)[0])
        if on_windows_workstation():
            if model_name == "microsoft/phi-2":
                model_path = Path.joinpath(TORCH_MODELS_PATH, "dolphin-2.6-phi-2")
            elif model_name == "mlabonne/phixtral-4x2_8":
                model_path = Path.joinpath(TORCH_MODELS_PATH, "phixtral-4x2_8")
                model_path = Path.joinpath(TORCH_MODELS_PATH, "phixtral-4x2_8-gptq-4bit-32g-actorder_True")
            elif model_name == "mistralai/Mixtral-8x7B-v0.1":
                model_path = Path.joinpath(TORCH_MODELS_PATH, "Mixtral-8x7B-v0.1")
            else:
                raise NotImplementedError(
                    "The selected model was not implemented or downloaded yet. Please choose another one."
                )

            # Load the model and tokenizer from the local path where the downloaded model is saved
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                # trust_remote_code: Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                # should only be set to `True` for repositories you trust and in which you have read the code, as it will
                # execute code present on the Hub on your local machine.
                trust_remote_code=True,
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=hardware_for_os(hardware_mode)[1],
                max_new_tokens=max_output_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                # device=0,
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=hardware_for_os(hardware_mode)[1],
                max_new_tokens=max_output_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError("Invalid abstraction mode for the OS.")

    return llm


def start_llamafile(model_name: str, ctx_size: int = 512) -> tuple[subprocess.Popen, TextIOWrapper]:
    """
    Start a llamafile server of the given model. Depending on the model a different port is used for the server.
    """
    if ctx_size > MAX_CTX_SIZE:
        raise ValueError(
            f"The given context size of {ctx_size} might be to large for the model. Please check the model details and increase the {MAX_CTX_SIZE} if your model allows a larger context."
        )
    model_name_l, output_file = output_file_path_for_model(model_name)
    if not os.path.exists(output_file):
        with open(output_file, "w") as mkfile:
            pass

    file = open(output_file, "r")

    llamafile_commands = {
        "mistral": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "mistral-7b-instruct-v0.1.Q5_K_M.gguf"))
        + " -ngl 9999",
        "mixtral": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf")),
        "codellama": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "codellama-70b-hf.Q5_K_M.gguf")),
        "dolphin-2.6-mistral": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "dolphin-2.6-mistral-7b.Q5_K_M.gguf"))
        + " -ngl 9999",
        "dolphin-2.7-mixtral": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf")),  # + " -ngl 12",
        "dolphincoder-starcoder2-15b": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "dolphincoder-starcoder2-15b.Q5_K_M.gguf"))
        + " -ngl 9999",
        "dolphin-2.6-phi-2": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "dolphin-2_6-phi-2.Q6_K.gguf"))
        + " -ngl 9999",
        "llama3": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"))
        + " -ngl 9999",
        "phi3": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "Phi-3-mini-4k-instruct-Q5_K_M.gguf"))
        + " -ngl 9999",
        "codestral": f"{str(LLAMAFILE_PATH)} --server --nobrowser -m "
        + str(Path.joinpath(GGUF_PATH, "Codestral-22B-v0.1-Q5_K_M.gguf")),
    }

    if model_name_l not in llamafile_commands.keys():
        raise NotImplementedError(
            f"The model you are trying to use is not available with llamafile. Model: {model_name}"
        )
    ctx_size = LLAMAFILE_CTX_SIZE[model_name_l]
    port = LLAMAFILE_PORTS[model_name_l]
    cmd = f"{llamafile_commands[model_name_l]} --port {port} -c {ctx_size} > {str(output_file)} 2>&1"
    print("Opening process with command:", cmd)
    process = subprocess.Popen(cmd, shell=True)
    time.sleep(1)
    return process, file


def output_file_path_for_model(model_name: str):
    model_name_l = model_name.lower()
    # ./llamafile_output/MODELNAME.txt
    output_file = Path.joinpath(LLAMAFILE_OUTPUT_LOG, model_name_l + ".txt")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    return model_name_l, output_file


def get_llamafile_port_for_model(model_name: str):
    model_name = model_name.lower()
    if model_name not in LLAMAFILE_PORTS.keys():
        raise NotImplementedError(
            f"The model you are trying to use is not available with llamafile. Model: {model_name}"
        )
    return LLAMAFILE_PORTS[model_name]


def check_llamafile_status(process: subprocess.Popen, model_name: str):
    """
    Check the status of the llamafile process for the given model.
    """
    poll = process.poll()
    match poll:
        case None:
            status = "running"
            health = llamafile_server_health(get_llamafile_port_for_model(model_name))
            if health is None:
                # MAYBE: wierd case, does this happen?
                return "no llamafile server"
            return f"{status} and {health}"
        case 0:
            # This probably does not exist
            return "finished"
        case 1:
            return "terminated"
        case _:
            return f"returncode: {poll}"


def llamafile_server_for_model_exists(model_name: str):
    """
    Check if a llamafile server for the given model exists on the respective port specified by `LLAMAFILE_PORTS`.
    """
    port = get_llamafile_port_for_model(model_name)
    res = utils.check_is_port_in_use(port)
    print(f"Port {port} is in use:", res)
    return res
    # MAYBE:
    # if utils.check_is_port_in_use(LLAMAFILE_PORTS[model_name]):
    #     return llamafile_server_health(LLAMAFILE_PORTS[model_name]) is not None
    # else:
    #     return False


def llamafile_server_health(port: int):
    """
    Wrapper for the llamafile server /health API endpoint.

    This fumction is only available with llamafile version >= 0.6.2.

    GET /health: Returns the current state of the server:
    - {"status": "loading model"} if the model is still being loaded.
    - {"status": "error"} if the model failed to load.
    - {"status": "ok"} if the model is successfully loaded and the server is ready for further requests.

    returns None if the request fails or does not fit the llamafile API
    """
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=30)

        response_json = response.json()
        if response.status_code in [200, 500, 503]:
            return response_json["status"]
        response.raise_for_status()
    # If the request fails (404) then print the error.
    except requests.exceptions.HTTPError as error:
        print(error)
    except requests.ConnectionError as error:
        print(error)
    return None


def llamafile_tokenization(port: int, text: str) -> list[int] | None:
    """
    Wrapper for the llamafile server /tokenize API endpoint.


    POST /tokenize: Returns the tokenized text as the integer IDs for the tokens.
    """
    request = {"content": text}
    try:
        response = requests.post(f"http://127.0.0.1:{port}/tokenize", json=request, timeout=20)

        response_json = response.json()
        if response.status_code in [200, 500, 503]:
            return response_json["tokens"]
        response.raise_for_status()
    # If the request fails (404) then print the error.
    except requests.exceptions.HTTPError as error:
        print(error)
    except requests.ConnectionError as error:
        print(error)
    return None


def simple_llamafile_server_ready(model_name: str):
    _, file_path = output_file_path_for_model(model_name)
    ready_strings = [
        '"level":"INFO","function":"server_cli","line":3289,"message":"HTTP server listening","port":"8092","hostname":"127.0.0.1"}',
        '"level":"INFO","function":"log_server_request","line":2741,"message":"request","remote_addr":"","remote_port":-1,"status":200,"method":"GET","path":"/index.js","params":{}}"',
    ]
    with open(file_path, "r") as file:
        text = file.read()
        for pattern in ready_strings:
            if pattern not in text:
                print(f"server for {model_name} NOT ready")
                return False
        print(f"server for {model_name} READY")
        return True


def kill_llamafile_server(process: subprocess.Popen, outfile: TextIOWrapper, model_name: str = ""):
    """
    Stop the llamafile server for with the given process and model name.
    """
    ret = process.kill()
    outfile.close()
    if ret is None:
        print("returncode:", ret)
    print(f"Killed the llamafile server process for the model: {model_name}.")
