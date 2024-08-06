from pathlib import Path
from codetrans.llm_abstraction import LLMSettings
import yaml


class TranslationMetadata:
    """Metadata for source code translations between programming languages"""

    def __init__(self, llm_name: str, llm_engine: str, llm_settings: LLMSettings, template_types: list[str]):
        self.llm_name = llm_name
        self.llm_engine = llm_engine
        self.llm_settings = llm_settings
        self.template_types = template_types

    def to_dict(self):
        metadata_dict = {
            "llm_name": self.llm_name,
            "llm_engine": self.llm_engine,
            "template_types": self.template_types,
        }
        metadata_dict.update(vars(self.llm_settings))
        return metadata_dict

    def save_to_file(self, file_path):
        """Saving the metadata to a file in the yaml format"""

        metadata_dict = self.to_dict()

        with open(file_path, "w") as f:
            yaml.dump(metadata_dict, f)

    def print_to_stdout(self):
        metadata_dict = self.to_dict()
        yaml_str = yaml.dump(metadata_dict)
        print(yaml_str)


def load_tranlation_metadata(file_path: Path | str) -> TranslationMetadata:
    with open(file_path, "r") as f:
        metadata_dict = yaml.load(f)
    return TranslationMetadata(
        metadata_dict["llm_name"],
        metadata_dict["llm_engine"],
        LLMSettings(
            metadata_dict["top_k"],
            metadata_dict["top_p"],
            metadata_dict["temperature"],
            metadata_dict["repeat_penalty"],
        ),
        metadata_dict["template_types"],
    )


if __name__ == "__main__":

    tm = TranslationMetadata(
        "mistral", "llamafile", LLMSettings(top_k=50, top_p=0.95, temperature=0.7, repeat_penalty=1.1), ["LIT"]
    )
    print(tm.to_dict())

    tm.print_to_stdout()
