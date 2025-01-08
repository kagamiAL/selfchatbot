import argparse
import json
import os.path as osp
import torch
import Util.utilities as U
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from os import environ as env
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Conv1D,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from Classes.FineTuners.fine_tuner import FineTuner
from Classes.Datasets.chat_dataset import ChatDataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from dotenv import load_dotenv
from Classes.Formatters.formatter import get_formatter


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
                    "max_length": parameters["max_length"],
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
    # User can specify the config for fine-tuning
    # If not specified, use the default config
    if "config" in parameters:
        return AutoConfig.from_pretrained(
            parameters["model"],
            **parameters["config"],
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


def get_model(parameters: dict, tokenizer: PreTrainedTokenizer) -> AutoModelForCausalLM:
    """Returns the model for fine-tuning

    Args:
        parameters (dict): parameters for fine-tuning
        tokenizer (PreTrainedTokenizer): tokenizer for fine-tuning

    Returns:
        AutoModelForCausalLM: model for fine-tuning
    """
    model: PreTrainedModel
    if parameters["type_fine_tune"] == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            parameters["model"], quantization_config=bnb_config
        )
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            parameters["model"],
            config=get_config(parameters),
        )
        # Resize the model embeddings if necessary
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
    if parameters["type_fine_tune"] in ["lora", "qlora"]:
        return get_peft_model(model, get_lora_config(parameters, model))
    return model


def save_results_json_data(save_path: Path, parameters: dict):
    """Saves the parameters as debug.json and the default generation parameters
    to parameters.json in the given save_path
    """
    save_default_generation_params(save_path, parameters)
    with open(save_path.joinpath("debug.json"), "w") as f:
        json.dump(parameters, f, indent="\t", separators=(",", ": "))


def fine_tuning_loop(dataset_id: int):
    """Fine tune a model on a dataset

    Args:
        dataset_id (int): ID of the dataset
    """
    preprocessed_path, dataset_name = U.get_path_to_dataset_and_name(
        "selfChatBot_preprocessed", dataset_id
    )
    save_path = Path(osp.join(env["selfChatBot_results"], dataset_name))
    save_path.mkdir(parents=True, exist_ok=True)
    parameters = get_params(preprocessed_path)
    base_model_name: str = parameters["model"]
    # * Save parameters before I start modifying them
    save_results_json_data(save_path, parameters)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    formatter = get_formatter(base_model_name, tokenizer)
    if (
        formatter.add_special_tokens_to_tokenizer()
        and parameters["type_fine_tune"] == "qlora"
    ):
        # * Need to load resized model for qlora
        # ! I probably shouldn't be modifying the parameters directly here but it works I don't care
        parameters["model"] = U.get_resized_model_path(base_model_name, formatter)
    model = get_model(parameters, tokenizer)
    chat_dataset = ChatDataset(get_corpora(preprocessed_path), tokenizer)
    # TODO: This is just me learning how to make my own training loop, I will use huggingface's more advanced training loop later
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
    q_lora_args = (
        {
            "fp16": True,
            "gradient_accumulation_steps": parameters["gradient_accumulation_steps"],
        }
        if parameters["type_fine_tune"] == "qlora"
        else {}
    )
    finetuner = FineTuner(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_path=save_path,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **q_lora_args,
    )
    model.config.use_cache = False
    finetuner.fine_tune(parameters["epochs"])


def main():
    """Main function to fine tune a model on a dataset

    Raises:
        Exception: If selfChatBot_preprocessed variable is not set in environment
        Exception: If selfChatBot_results variable is not set in environment
    """
    load_dotenv()
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
