import re
import argparse
import os
import os.path as osp
from os import environ as env
import json
from typing import Optional

preprocessors = {}


def get_path_to_dataset_and_name(dataset_id: int) -> tuple[str, str]:
    """Returns the path to the dataset with ID dataset_id

    Args:
        dataset_id (int): ID of the dataset

    Returns:
        tuple[str, str]: path to the dataset and name of the dataset
    """
    for dataset in os.listdir(env["selfChatBot_raw"]):
        result = re.findall(f"Dataset_\\d+", dataset)
        if result:
            if int(result[0].split("_")[1]) == dataset_id:
                return os.path.join(env["selfChatBot_raw"], dataset), dataset
    raise FileNotFoundError(f"Dataset with ID {dataset_id} not found")


def get_json_data(data_format_path: str) -> Optional[dict]:
    """Returns the parsed data.json file in data_format_path

    Args:
        data_format_path (str): path to the data format

    Returns:
        Optional[dict]: data.json file in data_format_path if it exists else None
    """
    json_path = osp.join(data_format_path, "data.json")
    if osp.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def preprocess_data(dataset_id: int):
    """Preprocesses data in dataset with ID dataset_id

    Args:
        dataset_id (int): ID of the dataset to preprocess
    """
    dataset_path, dataset_name = get_path_to_dataset_and_name(dataset_id)
    processed_text = []
    for data_format in os.listdir(dataset_path):
        if not data_format in preprocessors:
            continue
        format_path = osp.join(dataset_path, data_format)
        json_data = get_json_data(format_path)
        for file in os.listdir(format_path):
            if file.endswith(".json"):
                continue
            with open(osp.join(format_path, file), "r") as f:
                processed_text.append(preprocessors[data_format](f.read(), json_data))
    with open(
        osp.join(env["selfChatBot_preprocessed"], f"{dataset_name}.txt"),
        "w",
    ) as f:
        f.write("\n".join(processed_text))


def main():
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
    preprocess_data(args.d)


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
