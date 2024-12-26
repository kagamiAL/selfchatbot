import argparse
from os import environ as env
from Classes.ChatModels.chat_model import ChatModel
from DataProcessing.preprocess_data import get_path_to_dataset_and_name


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
    if not env.get("selfChatBot_results"):
        raise Exception("selfChatBot_results variable not set in environment")
    args = parser.parse_args()
    dataset_path, _ = get_path_to_dataset_and_name("selfChatBot_results", args.d)
    chat_model = ChatModel(dataset_path)
    match args.t:
        case "session":
            while True:
                prompt = input("You: ")
                response = chat_model.prompt_best_response(prompt, True)
                if response:
                    chat_model.add_to_history(prompt, response)
                else:
                    response = "<no response>"
                print("Response: " + response)
                print()
        case "prompt":
            if args.p is None:
                parser.error("Prompt interaction requires a prompt")
            print()
            for i, response in enumerate(chat_model.prompt_for_responses(args.p)):
                print(f"Response {i + 1}: {response}\n")
