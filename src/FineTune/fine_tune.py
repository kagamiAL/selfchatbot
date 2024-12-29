import argparse
import json
import os.path as osp
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from os import environ as env
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Conv1D,
    get_linear_schedule_with_warmup,
)
from Classes.FineTuners.fine_tuner import FineTuner
from Classes.Datasets.chat_dataset import ChatDataset
from DataProcessing.preprocess_data import get_path_to_dataset_and_name
from peft import get_peft_model, LoraConfig, TaskType


def get_specific_layer_names(model: AutoModelForCausalLM) -> list[str]:
    """Returns the names of the specific layers for lora fine-tuning

    Args:
        model (AutoModelForCausalLM): The model for fine-tuning

    Returns:
        list[str]: The names of the specific layers
    """
    layer_names = []

    valid = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, valid):
            # model name parsing
            layer = ".".join(name.split(".")[4:]).split(".")[0]
            if layer:
                layer_names.append(layer)
    return list(set(layer_names))


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_params(dataset_path: str) -> dict:
    """Returns the parameters for fine-tuning

    Args:
        dataset_path (str): path to the dataset

    Returns:
        dict: parameters for fine-tuning
    """
    with open(osp.join(dataset_path, "parameters.json"), "r") as f:
        return json.load(f)


def get_corpora(dataset_path: str) -> str:
    """Returns the corpora for fine-tuning

    Args:
        dataset_path (str): path to the dataset

    Returns:
        str: corpora for fine-tuning
    """
    with open(osp.join(dataset_path, "corpora.txt"), "r", encoding="utf-8") as f:
        return f.read()


def save_default_generation_params(save_path: Path, parameters: dict):
    """Saves the default generation parameters if they don't exist

    Args:
        save_path (Path): The save path
    """
    generation_params = save_path.joinpath("parameters.json")
    if not generation_params.is_file():
        with open(generation_params, "w") as f:
            json.dump(
                {
                    "model": parameters["model"],
                    "type_fine_tune": parameters["type_fine_tune"],
                    "max_length": 1024,
                    "temperature": 0.8,
                    "top_p": 0.85,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "num_return_sequences": 3,
                },
                f,
                indent="\t",
                separators=(",", ": "),
            )


def get_config(parameters: dict) -> AutoConfig:
    """Returns the config for fine-tuning

    Args:
        parameters (dict): parameters for fine-tuning

    Returns:
        AutoConfig: config for fine-tuning
    """
    if "dropout" in parameters:
        return AutoConfig.from_pretrained(
            parameters["model"],
            resid_pdrop=parameters["dropout"],
            embd_pdrop=parameters["dropout"],
            attn_pdrop=parameters["dropout"],
        )
    return AutoConfig.from_pretrained(parameters["model"])


def get_lora_config(parameters: dict, base_model: AutoModelForCausalLM) -> LoraConfig:
    """Returns the lora config for lora fine-tuning

    Args:
        parameters (dict): parameters for lora

    Returns:
        LoraConfig: lora config
    """
    # Get the names of the specific layers, if specified else get them automatically
    target_modules = (
        "target_modules" in parameters
        and parameters["target_modules"]
        or get_specific_layer_names(base_model)
    )
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=parameters["lora_r"],
        lora_alpha=parameters["lora_alpha"],
        lora_dropout=parameters["lora_dropout"],
        target_modules=target_modules,
    )


def get_model(parameters: dict) -> AutoModelForCausalLM:
    """Returns the model for fine-tuning

    Args:
        parameters (dict): parameters for fine-tuning

    Returns:
        AutoModelForCausalLM: model for fine-tuning
    """
    model = AutoModelForCausalLM.from_pretrained(
        parameters["model"],
        config=get_config(parameters),
    )
    if parameters["type_fine_tune"] == "lora":
        return get_peft_model(model, get_lora_config(parameters, model))
    return model


def fine_tuning_loop(dataset_id: int):
    """Fine tune a model on a dataset

    Args:
        dataset_id (int): ID of the dataset
    """
    dataset_path, dataset_name = get_path_to_dataset_and_name(
        "selfChatBot_preprocessed", dataset_id
    )
    parameters = get_params(dataset_path)
    model = get_model(parameters)
    chat_dataset = ChatDataset(
        get_corpora(dataset_path),
        AutoTokenizer.from_pretrained(parameters["model"]),
    )
    train_size = int(parameters["dataset_split"] * len(chat_dataset))
    val_size = len(chat_dataset) - train_size
    train_dataset, val_dataset = random_split(chat_dataset, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=parameters["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=parameters["batch_size"], shuffle=False
    )
    total_steps = len(train_dataloader) * parameters["epochs"]
    optimizer = AdamW(
        model.parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(parameters["warmup_steps_percent"] * total_steps),
        num_training_steps=total_steps,
    )
    save_path = Path(osp.join(env["selfChatBot_results"], dataset_name))
    save_path.mkdir(parents=True, exist_ok=True)
    save_default_generation_params(save_path, parameters)
    with open(save_path.joinpath("debug.json"), "w") as f:
        json.dump(parameters, f, indent="\t", separators=(",", ": "))
    finetuner = FineTuner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    finetuner.fine_tune(parameters["epochs"])


def main():
    """Main function to fine tune a model on a dataset

    Raises:
        Exception: If selfChatBot_preprocessed variable is not set in environment
        Exception: If selfChatBot_results variable is not set in environment
    """
    parser = argparse.ArgumentParser(description="Fine tune a model on a dataset")
    parser.add_argument(
        "-d",
        type=int,
        help="The ID of the dataset to fine tune on",
        required=True,
    )
    if not env.get("selfChatBot_preprocessed"):
        raise Exception("selfChatBot_preprocessed variable not set in environment")
    if not env.get("selfChatBot_results"):
        raise Exception("selfChatBot_results variable not set in environment")
    fine_tuning_loop(parser.parse_args().d)
