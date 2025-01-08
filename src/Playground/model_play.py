import torch
import argparse
import json
import Util.utilities as U
from os import environ as env
from pathlib import Path
from Classes.ChatModels.chat_model import ChatModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from Classes.Formatters.formatter import get_formatter
from dotenv import load_dotenv


def get_json_data(json_path: Path) -> dict:
    """Returns the parsed data.json file in data_format_path

    Args:
        data_format_path (str): path to the data format

    Returns:
        dict: parsed data.json file
    """
    with open(json_path, "r") as f:
        return json.load(f)


def get_chat_model_arguments(model_path: str, args: argparse.Namespace) -> tuple:
    """Returns the arguments to use for the chat model

    Args:
        model_path (str): The path to the model
        args (argparse.Namespace): The arguments from the command line

    Returns:
        tuple: The arguments to use for the chat model, (model, tokenizer, params)
    """

    # Define paths and load parameters
    parameters_path = Path(model_path) / "parameters.json"
    model_type_path = Path(model_path) / args.mt
    parameters = get_json_data(parameters_path)
    base_model_name: str = parameters["model"]

    # Initialize tokenizer and formatter
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    formatter = get_formatter(base_model_name, tokenizer)

    # Prepare model kwargs and check for resizing
    model_kwargs = {"device_map": "auto"}
    is_tokenizer_resized: bool = formatter.add_special_tokens_to_tokenizer()

    # Adjust parameters for QLoRA if needed
    if parameters["type_fine_tune"] == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if is_tokenizer_resized:
            parameters["model"] = U.get_resized_model_path(base_model_name, formatter)

    # Load the model based on fine-tuning type
    if parameters["type_fine_tune"] in ["lora", "qlora"]:
        # Load base model or resized model if lora or qlora
        model = AutoModelForCausalLM.from_pretrained(
            parameters["model"], **model_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_type_path, **model_kwargs)

    # Resize token embeddings if required
    if is_tokenizer_resized and parameters["type_fine_tune"] != "qlora":
        model.resize_token_embeddings(len(formatter.tokenizer))

    # Apply PEFT for LoRA or QLoRA fine-tuning
    if parameters["type_fine_tune"] in ["lora", "qlora"]:
        model = PeftModel.from_pretrained(model, model_type_path)

    return model, formatter, parameters


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Play with a chat model")
    parser.add_argument(
        "-d", type=int, help="The ID of the dataset to play on", required=True
    )
    parser.add_argument(
        "-t",
        default="session",
        const="session",
        nargs="?",
        choices=["session", "prompt"],
        help="The type of interaction to have with the model",
    )
    parser.add_argument(
        "-p",
        type=str,
        help="The prompt to use when the type of interaction is prompt",
    )
    parser.add_argument(
        "-mt",
        default="best",
        const="best",
        nargs="?",
        choices=["best", "final"],
        help="The model to use (best or final) to use for interaction",
    )
    if not env.get("selfChatBot_results"):
        raise Exception("selfChatBot_results variable not set in environment")
    args = parser.parse_args()
    dataset_path, _ = U.get_path_to_dataset_and_name("selfChatBot_results", args.d)
    chat_model = ChatModel(*get_chat_model_arguments(dataset_path, args))
    match args.t:
        case "session":
            print("Start a conversation with the model\n")
            while True:
                prompt = input("You: ")
                response = chat_model.prompt_best_response(prompt, True)
                if response:
                    chat_model.add_to_history(prompt, response)
                else:
                    response = "<no response>"
                print()
                print("\n\n".join([f"Model: {resp}" for resp in response.splitlines()]))
                print()
        case "prompt":
            if args.p is None:
                parser.error("Prompt interaction requires a prompt")
            print()
            for i, response in enumerate(chat_model.prompt_for_responses(args.p)):
                print(f"Response {i + 1}: {response}\n")
