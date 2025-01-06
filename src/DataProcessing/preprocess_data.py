import re
import argparse
import os
import os.path as osp
from os import environ as env
import json
from pathlib import Path
from Classes.Preprocessors.preprocessor import Preprocessor
from Classes.Schema import validate_json, schemas
from Classes.Formatters.formatter import get_formatter
from transformers import AutoTokenizer
from dotenv import load_dotenv

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


def get_preprocess_parameters(dataset_path: str) -> dict:
    """Returns preprocessing parameters from a JSON file.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        dict: Preprocessing parameters loaded from 'parameters.json'.
    """
    return validate_json.get_validated_json(
        Path(dataset_path).joinpath("parameters.json"),
        schemas.preprocess_parameters_schema,
    )


def generate_default_params(
    model_name: str, type_fine_tune: str, max_length: int
) -> dict:
    """Generate default parameters for fine-tuning

    Args:
        model_name (str): name of the model

    Returns:
        dict: default parameters
    """
    params = {
        "model": model_name,
        "type_fine_tune": type_fine_tune,
        "max_length": max_length,
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
        params["batch_size"] = 1
        params["gradient_accumulation_steps"] = 8
    return params


def validate_preprocessor_data(
    preprocessor_class: Preprocessor, preprocess_parameters: dict
):
    """
    Validates the preprocessor data in preprocess_parameters against the schema in the given preprocessor_class

    Args:
        preprocessor_class (Preprocessor): class of the preprocessor
        preprocess_parameters (dict): parameters for preprocessing

    Raises:
        ValueError: If the preprocessor data for the given data format is not found in parameters.json
    """
    if preprocessor_class.PREPROCESSOR_DATA_SCHEMA:
        data_format = preprocessor_class.DATA_FORMAT
        if not data_format in preprocess_parameters["preprocessor_data"]:
            raise ValueError(
                f"Preprocessor data for {data_format} not found in parameters.json"
            )
        validate_json.validate_dict(
            preprocess_parameters["preprocessor_data"][data_format],
            preprocessor_class.PREPROCESSOR_DATA_SCHEMA,
        )


def preprocess_data(args: argparse.Namespace):
    """Preprocesses data in dataset with ID dataset_id

    Args:
        dataset_id (int): ID of the dataset to preprocess
    """
    dataset_id = args.d
    # Get the path to the dataset and the name of the dataset
    dataset_path, dataset_name = get_path_to_dataset_and_name(
        "selfChatBot_raw", dataset_id
    )
    preprocess_parameters = get_preprocess_parameters(dataset_path)
    processed_text = []
    tokenizer = AutoTokenizer.from_pretrained(preprocess_parameters["model"])
    formatter = get_formatter(preprocess_parameters["model"], tokenizer)
    for data_format in os.listdir(dataset_path):
        if not data_format in preprocessors:
            continue
        preprocessor_class: Preprocessor = preprocessors[data_format]
        validate_preprocessor_data(preprocessor_class, preprocess_parameters)
        format_path = osp.join(dataset_path, data_format)
        preprocessor: Preprocessor = preprocessor_class(
            tokenizer, formatter, preprocess_parameters
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
            generate_default_params(
                preprocess_parameters["model"],
                preprocess_parameters["type_fine_tune"],
                preprocess_parameters["max_length"],
            ),
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
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Preprocesses data from different sources contained in a Dataset"
    )
    parser.add_argument(
        "-d",
        type=int,
        help="The ID of the dataset to preprocess",
        required=True,
    )
    args = parser.parse_args()
    if not env.get("selfChatBot_raw"):
        raise Exception("selfChatBot_raw variable not set in environment")
    if not env.get("selfChatBot_preprocessed"):
        raise Exception("selfChatBot_preprocessed variable not set in environment")
    preprocess_data(args)
