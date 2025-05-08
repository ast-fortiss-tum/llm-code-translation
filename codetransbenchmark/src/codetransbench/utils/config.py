from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class Config:
    base_dir: Path
    output_dir: Path
    dataset_dir: Path
    config_dir: Path
    logs_dir: Path
    testresults_dir: Path
    postprocessing_reports_dir: Path
    temp_exec_dir: Path
    cleaned_dir_name: str = "cleaned"


def load_config(file_str: str = ".\codetransbenchmark\config\config.yaml") -> Config:
    with open(file_str, "r") as f:
        config_dict = yaml.safe_load(f)

    print(f"Loaded config from file {file_str}: config: {config_dict}")

    base_dir = Path(config_dict["base_dir"])
    if base_dir == "set the absolute path to codetransbenchmark":
        raise Exception(
            "set the absolute path to codetransbenchmark in config/config.yaml"
        )

    return Config(
        base_dir=base_dir,
        output_dir=base_dir / str(config_dict["output_dir"]),
        dataset_dir=base_dir / str(config_dict["dataset_dir"]),
        config_dir=base_dir / str(config_dict["config_dir"]),
        logs_dir=base_dir / str(config_dict["logs_dir"]),
        testresults_dir=base_dir / str(config_dict["testresults_dir"]),
        postprocessing_reports_dir=base_dir
        / str(config_dict["postprocessing_reports_dir"]),
        temp_exec_dir=base_dir / str(config_dict["temp_exec_dir"]),
    )
