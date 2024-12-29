from abc import ABC, abstractmethod
from transformers import AutoConfig

formatters: list["Formatter"] = []


class Formatter(ABC):

    MODEL_NAME: str

    model_label: str
    user_label: str

    def __init__(self, model_label: str, user_label: str, tokenizer):
        self.tokenizer = tokenizer
        self.model_label = model_label
        self.user_label = user_label

    def get_labels(self) -> list[str]:
        """Returns the labels in the format [model_label, user_label]

        Returns:
            list[str]: [model_label, user_label]
        """
        return [self.model_label, self.user_label]

    @abstractmethod
    def format(self, text: str) -> str:
        """Formats a piece of text to a finetuneable string
        Text is in the form of
        "label text"

        Args:
            text (str): the text to format

        Raises:
            NotImplementedError

        Returns:
            str: the formatted text
        """
        raise NotImplementedError

    def format_block(self, block: list[str]) -> str:
        """Formats a block to a finetuneable string
        Blocks are in the form of
        [
            "label text",
        ]

        Args:
            block (list[str]): the block to format

        Returns:
            str: the formatted block
        """
        return "".join(self.format(line) for line in block)


def get_formatter(model_name: str, tokenizer) -> Formatter:
    """Returns a formatter for a given model name

    Args:
        model_name (str): The model name to get the formatter for
        tokenizer (transformers.AutoTokenizer): The tokenizer to pass to the formatter

    Returns:
        Formatter: The formatter for the given model name

    Raises:
        ValueError: If no formatter is found for the given model name
    """
    if model_name in formatters:
        return formatters[model_name](tokenizer)
    config = AutoConfig.from_pretrained(model_name)
    architectures = getattr(config, "architectures", None)
    if architectures:
        for architecture in architectures:
            if architecture in formatters:
                return formatters[architecture](tokenizer)
    raise ValueError(f"No formatter found for model {model_name}")


def register_formatter(model_name: str, formatter: Formatter):
    """Registers a formatter for a given model name.

    Args:
        model_name (str): The model name to register the formatter for.
        formatter (Formatter): The formatter to register.

    """
    formatters[model_name] = formatter
