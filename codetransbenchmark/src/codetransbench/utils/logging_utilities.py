import os
import logging
import logging.config

from codetransbench.utils.config import Config


def setup_logging(task, model, config: Config):
    """Load logging configuration"""

    log_configs = {"simple": "logging.ini"}
    c = log_configs.get("simple", "logging.ini")
    config_path = config.config_dir / c

    logs_dir_parts = config.logs_dir.parts
    logs_dir_parts = [x.replace("\\", "") for x in logs_dir_parts]
    logs_directory = "/".join(list(logs_dir_parts))
    normal = "/".join([logs_directory, f"{task}_{model}.log"])
    warnings = "/".join([logs_directory, f"{task}_{model}_errors.log"])
    print(normal, warnings)

    logging.config.fileConfig(
        config_path,
        disable_existing_loggers=False,
        defaults={"logfilename": normal, "errorfilename": warnings},
    )
