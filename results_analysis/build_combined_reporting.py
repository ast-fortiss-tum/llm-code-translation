import os
import pandas as pd
import json
from pathlib import Path
from codetransbench.utils.dataset_information import language_pairs
from codetransbench.utils.config import load_config, Config

from codetransbench.translation.translate_open_source import FILE_EXTENSIONS

reports_dir = "./reports"

INFINITE_LOOP_CATEGORY = "incorrect"


def create_exec_report(
    datasets: list[str], model_name: str, template: str, max_attempts: int, config: Config, opp: str | None = None
):
    if opp:
        testresults_dir = config.testresults_dir.parent / f"{config.testresults_dir.stem}_{opp}_pp"
    else:
        testresults_dir = config.testresults_dir
    dfs = {}
    for ds in datasets:

        success_report = []

        for source_pl, target_pl in language_pairs(ds):
            successful = []
            input_source_dir = config.dataset_dir / ds / source_pl / "Code"
            for attempt in range(1, max_attempts + 1):
                translations_dir = (
                    config.output_dir / f"{model_name}_{template}" / f"iteration_{attempt}" / ds / source_pl / target_pl
                )
                report_dir = testresults_dir / model_name / template

                if not (os.path.exists(translations_dir) or os.path.exists(report_dir)):
                    # print(translations_dir, "does not exist")
                    break

                # ordered_unsuccessful of language pair
                ordered_unsuccessful_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_compileReport_from_{source_pl}_to_{target_pl}_ordered_unsuccessful.txt"
                )
                if not (os.path.exists(ordered_unsuccessful_fp)):
                    # print(ordered_unsuccessful_fp, "does not exist")
                    break
                ordered_unsuccessful = []
                with open(ordered_unsuccessful_fp, "r") as f:
                    ordered_unsuccessful = f.read().splitlines()

                # dataset source files faulty
                faulty_tasks_fp = (
                    testresults_dir
                    / "llamafile_x"
                    / "test_source"
                    / f"iteration_1"
                    / f"llamafile_x_{ds}_compileReport_from_{source_pl}_to_{source_pl}_ordered_unsuccessful.txt"
                )
                faulty_tasks = []
                if not (os.path.exists(faulty_tasks_fp)):
                    # print(f"{faulty_tasks_fp} does not exist")
                    pass
                else:
                    with open(faulty_tasks_fp, "r") as f:
                        faulty_tasks = f.read().splitlines()

                # error_report

                error_report = {}
                json_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_errors_from_{source_pl}_to_{target_pl}_{attempt}.json"
                )
                with open(json_fp, "r") as f:
                    error_report: dict = json.load(f)

                for file in os.listdir(input_source_dir):
                    # print(file)
                    name, _ = os.path.splitext(file)
                    target_file = name + f".{FILE_EXTENSIONS[target_pl]}"

                    if not str(file).endswith(FILE_EXTENSIONS[source_pl]) or target_file in successful:
                        continue

                    file_information = {
                        "source_filename": file,
                        "target_filename": target_file,
                        "dataset": ds,
                        "source_lang": source_pl,
                        "target_lang": target_pl,
                        "model_name": model_name,
                        "template": template,
                        "attempt": attempt,
                    }

                    if file in faulty_tasks:
                        # the source code file in the dataset is faulty (compile/runtime/test error)
                        file_information[f"result"] = "faulty source"

                    elif target_file not in os.listdir(translations_dir):
                        # unicode error
                        file_information["result"] = "unicode error"
                    elif target_file in ordered_unsuccessful:
                        found = False
                        for error, failed_files in error_report.items():
                            if found:
                                break
                            for entry in failed_files:
                                if isinstance(entry, str):
                                    filename = entry
                                else:
                                    filename = entry[0]
                                if filename == target_file:
                                    if ((error == "runtime") and (filename == entry)) or ("infinite loop" in entry[1]):
                                        file_information["result"] = INFINITE_LOOP_CATEGORY  # "infinite loop"
                                    else:
                                        file_information["result"] = error
                                    found = True
                                    break

                    else:
                        # not already in successful / unicode error / unsuccessful
                        successful.append(target_file)
                        file_information["result"] = "success"
                    success_report.append(file_information)

        df = pd.DataFrame.from_dict(success_report)

        os.makedirs("./reports", exist_ok=True)

        if df.shape != (0, 0):
            df.to_excel(f"./reports/exec_report_{model_name}_{template}_{ds}.xlsx")
            if opp:
                df.to_excel(f"./reports/exec_report_{model_name}_{template}_{ds}_{opp}.xlsx")
            else:
                df.to_excel(f"./reports/exec_report_{model_name}_{template}_{ds}_{ds}.xlsx")
            dfs[ds] = df

    return success_report, dfs


def create_exec_flow_report_pp_naive(
    datasets: list[str], model_name: str, template: str, max_attempts: int, config: Config, opp: str | None = None
) -> tuple[dict, dict[pd.DataFrame]]:
    if opp:
        testresults_dir = config.testresults_dir.parent / f"{config.testresults_dir.stem}_{opp}_pp"
    else:
        testresults_dir = config.testresults_dir

    dfs = dict()
    for ds in datasets:
        success_report = dict()

        for source_pl, target_pl in language_pairs(ds):
            successful = []
            input_source_dir = config.dataset_dir / ds / source_pl / "Code"
            for attempt in range(1, max_attempts + 1):
                translations_dir = (
                    config.output_dir / f"{model_name}_{template}" / f"iteration_{attempt}" / ds / source_pl / target_pl
                )
                report_dir = testresults_dir / model_name / template

                if not os.path.exists(report_dir):
                    print(report_dir, "does not exist")
                    break

                compile_report = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_compileReport_from_{source_pl}_to_{target_pl}.txt"
                )
                if not os.path.exists(compile_report):
                    print(compile_report, "does not exist")
                    break

                with open(compile_report, "r") as f:
                    compile_report = f.read().splitlines()

                for line in compile_report:
                    if "Successfull Test Files: " in line:
                        line = line.replace("]", "")
                        line = line.replace("'", "")
                        successful_part = line.split(": [")[1].split(", ")
                        break

                # ordered_unsuccessful of language pair
                ordered_unsuccessful_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_compileReport_from_{source_pl}_to_{target_pl}_ordered_unsuccessful.txt"
                )
                if not (os.path.exists(ordered_unsuccessful_fp)):
                    print(ordered_unsuccessful_fp, "does not exist")
                    break
                ordered_unsuccessful = []
                with open(ordered_unsuccessful_fp, "r") as f:
                    ordered_unsuccessful = f.read().splitlines()

                # dataset source files faulty
                faulty_tasks_fp = (
                    testresults_dir
                    / "llamafile_x"
                    / "test_source"
                    / f"iteration_1"
                    / f"llamafile_x_{ds}_compileReport_from_{source_pl}_to_{source_pl}_ordered_unsuccessful.txt"
                )
                faulty_tasks = []
                if not (os.path.exists(faulty_tasks_fp)):
                    # print(f"{faulty_tasks_fp} does not exist")
                    pass
                else:
                    with open(faulty_tasks_fp, "r") as f:
                        faulty_tasks = f.read().splitlines()

                # error_report

                error_report = {}
                json_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_errors_from_{source_pl}_to_{target_pl}_{attempt}.json"
                )
                with open(json_fp, "r") as f:
                    error_report: dict = json.load(f)

                for file in os.listdir(input_source_dir):
                    # print(file)
                    name, _ = os.path.splitext(file)
                    target_file = name + f".{FILE_EXTENSIONS[target_pl]}"

                    if not str(file).endswith(FILE_EXTENSIONS[source_pl]) or target_file in successful:
                        continue

                    file_id = f"{model_name}_{template}_{ds}_{file}_{target_file}_{source_pl}_{target_pl}"

                    file_information = {
                        "source_filename": file,
                        "target_filename": target_file,
                        "dataset": ds,
                        "source_lang": source_pl,
                        "target_lang": target_pl,
                        "model_name": model_name,
                        "template": template,
                    }

                    file_information = success_report.get(file_id, file_information)

                    if file in faulty_tasks:
                        # the source code file in the dataset is faulty (compile/runtime/test error)
                        file_information[f"result_{attempt}"] = "faulty source"

                    # elif target_file not in os.listdir(translations_dir):
                    #     # unicode error
                    #     file_information[f"result_{attempt}"] = "unicode error"

                    elif target_file in ordered_unsuccessful:
                        found = False
                        for error, failed_files in error_report.items():
                            if found:
                                break
                            for entry in failed_files:
                                if isinstance(entry, str):
                                    filename = entry
                                else:
                                    filename = entry[0]
                                if filename == target_file:
                                    if ((error == "runtime") and (filename == entry)) or ("infinite loop" in entry[1]):
                                        file_information[f"result_{attempt}"] = (
                                            INFINITE_LOOP_CATEGORY  # "infinite loop"
                                        )
                                    else:
                                        file_information[f"result_{attempt}"] = error
                                    found = True
                                    break

                    elif target_file in successful_part:
                        # not already in successful / unicode error / unsuccessful / faulty source

                        successful.append(target_file)
                        file_information[f"result_{attempt}"] = "success"

                    else:
                        # target_file not in os.listdir(translations_dir):
                        # unicode error
                        file_information[f"result_{attempt}"] = "unicode error"
                    success_report[file_id] = file_information

        df = pd.DataFrame.from_dict(success_report.values())
        os.makedirs("./reports", exist_ok=True)

        if df.shape != (0, 0):
            if opp:
                df.to_excel(f"./reports/exec_mitigation_report_{model_name}_{template}_{ds}_{opp}.xlsx")
            else:
                df.to_excel(f"./reports/exec_mitigation_report_{model_name}_{template}_{ds}.xlsx")
            dfs[ds] = df

    return success_report, dfs


def create_exec_flow_report(
    datasets: list[str], model_name: str, template: str, max_attempts: int, config: Config, opp: str | None = None
) -> tuple[dict, dict[pd.DataFrame]]:
    if opp:
        testresults_dir = config.testresults_dir.parent / f"{config.testresults_dir.stem}_{opp}_pp"
    else:
        testresults_dir = config.testresults_dir
    dfs = dict()
    for ds in datasets:
        success_report = dict()

        for source_pl, target_pl in language_pairs(ds):
            successful = []
            input_source_dir = config.dataset_dir / ds / source_pl / "Code"
            for attempt in range(1, max_attempts + 1):
                translations_dir = (
                    config.output_dir / f"{model_name}_{template}" / f"iteration_{attempt}" / ds / source_pl / target_pl
                )
                report_dir = testresults_dir / model_name / template

                if not (os.path.exists(translations_dir) or os.path.exists(report_dir)):
                    # print(translations_dir, "does not exist")
                    break

                # ordered_unsuccessful of language pair
                ordered_unsuccessful_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_compileReport_from_{source_pl}_to_{target_pl}_ordered_unsuccessful.txt"
                )
                if not (os.path.exists(ordered_unsuccessful_fp)):
                    # print(ordered_unsuccessful_fp, "does not exist")
                    break
                ordered_unsuccessful = []
                with open(ordered_unsuccessful_fp, "r") as f:
                    ordered_unsuccessful = f.read().splitlines()

                # dataset source files faulty
                faulty_tasks_fp = (
                    testresults_dir
                    / "llamafile_x"
                    / "test_source"
                    / f"iteration_1"
                    / f"llamafile_x_{ds}_compileReport_from_{source_pl}_to_{source_pl}_ordered_unsuccessful.txt"
                )
                faulty_tasks = []
                if not (os.path.exists(faulty_tasks_fp)):
                    # print(f"{faulty_tasks_fp} does not exist")
                    pass
                else:
                    with open(faulty_tasks_fp, "r") as f:
                        faulty_tasks = f.read().splitlines()

                # error_report

                error_report = {}
                json_fp = (
                    report_dir
                    / f"iteration_{attempt}"
                    / f"{model_name}_{ds}_errors_from_{source_pl}_to_{target_pl}_{attempt}.json"
                )
                with open(json_fp, "r") as f:
                    error_report: dict = json.load(f)

                for file in os.listdir(input_source_dir):
                    # print(file)
                    name, _ = os.path.splitext(file)
                    target_file = name + f".{FILE_EXTENSIONS[target_pl]}"

                    if not str(file).endswith(FILE_EXTENSIONS[source_pl]) or target_file in successful:
                        continue

                    file_id = f"{model_name}_{template}_{ds}_{file}_{target_file}_{source_pl}_{target_pl}"

                    file_information = {
                        "source_filename": file,
                        "target_filename": target_file,
                        "dataset": ds,
                        "source_lang": source_pl,
                        "target_lang": target_pl,
                        "model_name": model_name,
                        "template": template,
                    }

                    file_information = success_report.get(file_id, file_information)

                    if file in faulty_tasks:
                        # the source code file in the dataset is faulty (compile/runtime/test error)
                        file_information[f"result_{attempt}"] = "faulty source"

                    elif target_file not in os.listdir(translations_dir):
                        # unicode error
                        file_information[f"result_{attempt}"] = "unicode error"

                    elif target_file in ordered_unsuccessful:
                        found = False
                        for error, failed_files in error_report.items():
                            if found:
                                break
                            for entry in failed_files:
                                if isinstance(entry, str):
                                    filename = entry
                                else:
                                    filename = entry[0]
                                if filename == target_file:
                                    if ((error == "runtime") and (filename == entry)) or ("infinite loop" in entry[1]):
                                        file_information[f"result_{attempt}"] = (
                                            INFINITE_LOOP_CATEGORY  # "infinite loop"
                                        )
                                    else:
                                        file_information[f"result_{attempt}"] = error
                                    found = True
                                    break

                    else:
                        # not already in successful / unicode error / unsuccessful / faulty source
                        successful.append(target_file)
                        file_information[f"result_{attempt}"] = "success"
                    success_report[file_id] = file_information

        df = pd.DataFrame.from_dict(success_report.values())
        os.makedirs("./reports", exist_ok=True)

        if df.shape != (0, 0):
            if opp:
                df.to_excel(f"./reports/exec_mitigation_report_{model_name}_{template}_{ds}_{opp}.xlsx")
            else:
                df.to_excel(f"./reports/exec_mitigation_report_{model_name}_{template}_{ds}.xlsx")
            dfs[ds] = df

    return success_report, dfs


def main_general(config: Config, opp: str | None = None):
    datasets = ["codenet", "avatar", "bithacks"]  # , "evalplus"]

    models = [
        "llamafile_mistral",
        "llamafile_mixtral",
        "llamafile_dolphin-2.6-mistral",
        "llamafile_dolphin-2.7-mixtral",
        "llamafile_dolphin-2.6-phi-2",
        "llamafile_phi3",
        "ollama_llama3-8b",
        "llamafile_codestral",
    ]
    templates = ["controlled", "controlled_md", "via_description", "LIT"]
    templates = ["controlled_md"] if opp else templates

    max_attempts = 1 if opp else 3
    df_list = []
    df_by_template = []
    df_by_model = []
    df_mitigation_raw = []

    for model_name in models:
        for template in templates:
            create_exec_report(datasets, model_name, template, max_attempts, config, opp)

            if opp:
                _, dfs = create_exec_flow_report_pp_naive(datasets, model_name, template, max_attempts, config, opp)
            else:
                _, dfs = create_exec_flow_report(datasets, model_name, template, max_attempts, config, opp)

            for df in dfs.values():
                df_mitigation_raw.append(df)
                df_list.append(
                    df[
                        [
                            "dataset",
                            "source_lang",
                            "target_lang",
                            "model_name",
                            "template",
                            "result_1",
                            "target_filename",
                        ]
                    ]
                    .groupby(
                        [
                            "dataset",
                            "model_name",
                            "template",
                            "source_lang",
                            "target_lang",
                            "result_1",
                        ]
                    )
                    .count()
                )
                df_by_template.append(
                    df[
                        [
                            "dataset",
                            "source_lang",
                            "target_lang",
                            "model_name",
                            "template",
                            "result_1",
                            "target_filename",
                        ]
                    ]
                    .groupby(["dataset", "template", "result_1"])
                    .count()
                )
                df_by_model.append(
                    df[
                        [
                            "dataset",
                            "source_lang",
                            "target_lang",
                            "model_name",
                            "template",
                            "result_1",
                            "target_filename",
                        ]
                    ]
                    .groupby(["dataset", "model_name", "result_1"])
                    .count()
                )
            # print(df_list)
    combined_raw_count = pd.concat(df_list)
    raw_combined = pd.concat(df_mitigation_raw)
    combined_by_template = pd.pivot_table(
        pd.concat(df_by_template).reset_index(),
        values="target_filename",
        index=["dataset", "model_name", "template", "source_lang", "target_lang"],
        columns=["result_1"],
        aggfunc=["sum"],
        fill_value=0,
        margins=True,
    )
    combined_by_model = pd.pivot_table(
        pd.concat(df_by_model).reset_index(),
        values="target_filename",
        index=["dataset", "model_name", "template", "source_lang", "target_lang"],
        columns=["result_1"],
        aggfunc=["sum"],
        fill_value=0,
        margins=True,
    )
    os.makedirs(Path("./data"), exist_ok=True)
    if opp:
        combined_raw_count.to_csv(Path("./data") / f"combined_exec_{opp}.csv")
        combined_by_model.to_csv(Path("./data") / f"combined_exec_by_model_{opp}.csv")
        combined_by_template.to_csv(Path("./data") / f"combined_exec_by_template_{opp}.csv")
        raw_combined.to_csv(Path("./data") / f"raw_combined_mitigation_{opp}.csv")
    else:
        combined_raw_count.to_csv(Path("./data") / "combined_exec.csv")
        combined_by_model.to_csv(Path("./data") / "combined_exec_by_model.csv")
        combined_by_template.to_csv(Path("./data") / "combined_exec_by_template.csv")
        raw_combined.to_csv(Path("./data") / "raw_combined_mitigation.csv")


if __name__ == "__main__":
    config = load_config("../codetransbenchmark/config/config.yaml")

    main_general(config)
