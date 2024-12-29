import argparse
import json
from os import environ as env
from pathlib import Path
from Classes.ChatModels.chat_model import ChatModel
from DataProcessing.preprocess_data import get_path_to_dataset_and_name
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    # base_path = Path(model_path)
    # self.history = []
    # self.model = AutoModelForCausalLM.from_pretrained(model_path)
    # self.model.eval()
    # self.tokenizer = AutoTokenizer.from_pretrained(
    #    get_json_data(base_path.joinpath("config.json"))["_name_or_path"]
    # )
    # self.generation_params = get_json_data(
    #    base_path.joinpath("generation_params.json")
    # )
    path: Path = Path(model_path)
    parameters: dict = get_json_data(path.joinpath("parameters.json"))
    model: AutoModelForCausalLM
    if parameters["type_fine_tune"] == "lora":
        model = AutoModelForCausalLM.from_pretrained(parameters["model"])
        model = PeftModel.from_pretrained(model, path.joinpath(args.mt))
    else:
        model = AutoModelForCausalLM.from_pretrained(path.joinpath(args.mt))
    return model, AutoTokenizer.from_pretrained(parameters["model"]), parameters


def main():
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
    dataset_path, _ = get_path_to_dataset_and_name("selfChatBot_results", args.d)
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
