"""collects the CLI arguments in an object"""

import argparse
from dataclasses import dataclass


@dataclass
class CLIArguments(argparse.Namespace):
    model: str
    dataset: str
    template_type: str


@dataclass
class CLIArgumentsTranslation(CLIArguments):
    source_lang: str
    target_lang: str
    k: int
    p: float
    temperature: float


@dataclass
class CLIArgumentsTranslationIteration(CLIArgumentsTranslation):
    attempt: int


@dataclass
class CLIArgumentsExecution(CLIArguments):
    source_lang: str
    target_lang: str
    report_dir: str
    attempt: int


@dataclass
class CLIArgumentsCleanGenerations(CLIArguments):
    attempt: int
    pp_type: str
