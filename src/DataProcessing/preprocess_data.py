import re
import argparse
import os
import os.path as osp
from os import environ as env
import json
from typing import Optional
from pathlib import Path
from Classes.Preprocessors.preprocessor import Preprocessor

preprocessors: dict[str, Preprocessor] = {}


def get_path_to_dataset_and_name(variable: str, dataset_id: int) -> tuple[str, str]:
    """Returns the path to the dataset with ID dataset_id

    Args:
        variable (str): name of the environment variable
        dataset_id (int): ID of the dataset

    Returns:
        tuple[str, str]: path to the dataset and name of the dataset
    """
    for dataset in os.listdir(env[variable]):
        result = re.findall(f"Dataset_\\d+", dataset)
        if result:
            if int(result[0].split("_")[1]) == dataset_id:
                return os.path.join(env[variable], dataset), dataset
    raise FileNotFoundError(f"Dataset with ID {dataset_id} not found")


def get_json_data(data_format_path: str, model: str) -> Optional[dict]:
    """Returns the parsed data.json file in data_format_path

    Args:
        data_format_path (str): path to the data format
        model (str): name of the model

    Returns:
        dict: data.json file in data_format_path
    """
    json_data = {}
    json_path = osp.join(data_format_path, "data.json")
    if osp.exists(json_path):
        with open(json_path, "r") as f:
            json_data = json.load(f)
    json_data["model"] = model
    return json_data


def generate_default_params(model_name: str, type_fine_tune: str) -> dict:
    """Generate default parameters for fine-tuning

    Args:
        model_name (str): name of the model

    Returns:
        dict: default parameters
    """
    params = {
        "model": model_name,
        "type_fine_tune": type_fine_tune,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "warmup_steps_percent": 0.05,
        "epochs": 500,
        "weight_decay": 0.01,
        "dataset_split": 0.8,
    }
    if type_fine_tune == "lora" or type_fine_tune == "qlora":
        params["lora_r"] = 32
        params["lora_alpha"] = 64
        params["lora_dropout"] = 0.05
    if type_fine_tune == "qlora":
        params["gradient_accumulation_steps"] = 8
        params["bit_quantization"] = 4
    return params


def preprocess_data(args: argparse.Namespace):
    """Preprocesses data in dataset with ID dataset_id

    Args:
        dataset_id (int): ID of the dataset to preprocess
    """
    dataset_id = args.d
    model = args.m
    dataset_path, dataset_name = get_path_to_dataset_and_name(
        "selfChatBot_raw", dataset_id
    )
    processed_text = []
    for data_format in os.listdir(dataset_path):
        if not data_format in preprocessors:
            continue
        format_path = osp.join(dataset_path, data_format)
        preprocessor: Preprocessor = preprocessors[data_format](
            get_json_data(format_path, model)
        )
        for file in os.listdir(format_path):
            if not file.endswith(".txt"):
                continue
            with open(osp.join(format_path, file), "r", encoding="utf-8") as f:
                processed_text.append(preprocessor.preprocess(f.read()))
    directory_path = osp.join(env["selfChatBot_preprocessed"], dataset_name)
    Path(directory_path).mkdir(exist_ok=True)
    with open(
        osp.join(directory_path, "corpora.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n\n".join(processed_text))
    with open(osp.join(directory_path, "parameters.json"), "w") as f:
        json.dump(
            generate_default_params(model, args.t),
            f,
            indent="\t",
            separators=(",", ": "),
        )


def add_preprocessor(name, func):
    """
    Add a new preprocessor

    Args:
        name (str): name of the preprocessor
        func (function): function to preprocess

    Returns:
        None
    """
    preprocessors[name] = func


def main():
    """Main function to preprocess data from different sources contained in a Dataset

    Raises:
        Exception: If selfChatBot_raw variable is not set in environment
        Exception: If selfChatBot_preprocessed variable is not set in environment
    """
    parser = argparse.ArgumentParser(
        description="Preprocesses data from different sources contained in a Dataset"
    )
    parser.add_argument(
        "-d",
        type=int,
        help="The ID of the dataset to preprocess",
        required=True,
    )
    parser.add_argument(
        "-m",
        type=str,
        help="The model you want to fine tune on",
        required=True,
    )
    parser.add_argument(
        "-t",
        default="lora",
        const="lora",
        nargs="?",
        choices=["finetune", "lora", "qlora"],
        help="The type of fine-tuning to use",
    )
    args = parser.parse_args()
    if not env.get("selfChatBot_raw"):
        raise Exception("selfChatBot_raw variable not set in environment")
    if not env.get("selfChatBot_preprocessed"):
        raise Exception("selfChatBot_preprocessed variable not set in environment")
    preprocess_data(args)
