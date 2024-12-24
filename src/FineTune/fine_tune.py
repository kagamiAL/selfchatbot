import argparse
import json
import os.path as osp
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from os import environ as env
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    GPT2Config,
    get_linear_schedule_with_warmup,
)
from FineTune.finetuner import FineTuner
from FineTune.chat_dataset import ChatDataset
from DataProcessing.preprocess_data import get_path_to_dataset_and_name


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


def fine_tuning_loop(dataset_id: int):
    """Fine tune a model on a dataset

    Args:
        dataset_id (int): ID of the dataset
    """
    dataset_path, dataset_name = get_path_to_dataset_and_name(
        "selfChatBot_preprocessed", dataset_id
    )
    parameters = get_params(dataset_path)
    dropout = parameters["dropout"]
    config = GPT2Config.from_pretrained(
        parameters["model"], resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout
    )
    model = GPT2LMHeadModel.from_pretrained(parameters["model"], config=config)
    chat_dataset = ChatDataset(
        get_corpora(dataset_path),
        GPT2TokenizerFast.from_pretrained(parameters["model"]),
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
    finetuner = FineTuner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset_name=dataset_name,
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
