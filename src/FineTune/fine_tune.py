import torch
import argparse
import json
import os.path as osp
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from os import environ as env
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
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
    device = torch.cuda.is_available() and "cuda" or "cpu"
    parameters = get_params(dataset_path)
    corpora = get_corpora(dataset_path)
    model = GPT2LMHeadModel.from_pretrained(parameters["model"]).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(parameters["model"])
    chat_dataset = ChatDataset(corpora, tokenizer)
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
    optimizer = AdamW(model.parameters(), lr=parameters["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(parameters["warmup_steps_percent"] * total_steps),
        num_training_steps=total_steps,
    )
    for i in range(parameters["epochs"]):
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        for step, batch in enumerate(train_dataloader):
            printProgressBar(
                step + 1,
                len(train_dataloader),
                prefix=f"Epoch: {
                    i + 1}",
                suffix=f"{step + 1}/{len(train_dataloader)}",
            )
            input_ids = batch[0].to(device)  # Input IDs (text data)
            labels = batch[0].to(device)  # Labels are the same as input_ids
            attn_mask = batch[1].to(device)  # Attention mask
            optimizer.zero_grad()  # Clear the gradients
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attn_mask,
            )
            loss = outputs.loss  # Extract loss from model's output
            total_train_loss += loss.item()
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model parameters
            scheduler.step()  # Step the learning rate scheduler
        # TODO: Calculate average validation loss for checkpointing


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
