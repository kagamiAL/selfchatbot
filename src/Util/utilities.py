import os
import re
from pathlib import Path
from os import environ as env
from transformers import AutoModelForCausalLM, PreTrainedModel
from Classes.Formatters.formatter import Formatter
from pathlib import Path
from shutil import rmtree


def clear_directory(directory_path: Path) -> None:
    """Clears the directory at the given path

    Args:
        directory_path (str): The path to the directory to clear
    """
    for path in directory_path.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


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


def get_resized_model_path(base_model: str, formatter: Formatter) -> str:
    """Returns the path to the resized model for the given base model and formatter

    Args:
        base_model (str): The name of the base model
        formatter (Formatter): The formatter to use for resizing the model

    Returns:
        str: The path to the resized model
    """
    resized_path: Path = (
        Path(env["selfChatBot_results"])
        .joinpath(".cache")
        .joinpath(f"{base_model}_{formatter.MODEL_NAME}")
    )
    resized_path.mkdir(parents=True, exist_ok=True)
    try:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(resized_path)
    except:
        print(f"Cached model does not exist or is corrupted, re-downloading")
        model = AutoModelForCausalLM.from_pretrained(base_model)
    if model.get_input_embeddings().num_embeddings != len(formatter.tokenizer):
        clear_directory(resized_path)
        model.resize_token_embeddings(len(formatter.tokenizer))
        model.save_pretrained(resized_path)
    return str(resized_path.absolute())
