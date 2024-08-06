# print(platform.system())

from pathlib import Path


LLAMAFILE_VERSION = "0.6.2"

TORCH_MODELS_PATH = Path("set this path")
"""Path to the base directory of the transformers and pytorch model files."""

LLAMAFILE_PATH = Path("set this path")
"""Path to the llamafile executable."""

GGUF_PATH = Path("set this path")
"""Path to the directory with the GGUF files of the models."""

LLAMAFILE_OUTPUT_LOG = Path("set this path")
"""Path to the directory where the llamafile output log files are stored."""
