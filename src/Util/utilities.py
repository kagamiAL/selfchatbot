import os
import re
import os.path as osp
from pathlib import Path
from os import environ as env
from transformers import AutoTokenizer, PreTrainedTokenizer


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


def get_tokenizer(path: str, model_name: str) -> PreTrainedTokenizer:
    """Returns the tokenizer to use based on if a custom tokenizer is found at path

    Args:
        path (str): The path to the preprocessed dataset
        model_name (str): The name of the model to use as a fallback

    Returns:
        PreTrainedTokenizer: The tokenizer to use
    """
    path = Path(path).joinpath("tokenizer")
    if path.is_dir():
        return AutoTokenizer.from_pretrained(path)
    return AutoTokenizer.from_pretrained(model_name)


def save_tokenizer(directory_path: str, tokenizer: PreTrainedTokenizer):
    """
    Saves a custom tokenizer to a specified directory.

    Args:
        directory_path (str): The path to the directory where the tokenizer should be saved.
        tokenizer (PreTrainedTokenizer): The tokenizer to be saved.
    """

    save_path = osp.join(directory_path, "tokenizer")
    Path(save_path).mkdir(exist_ok=True)
    tokenizer.save_pretrained(save_path)
