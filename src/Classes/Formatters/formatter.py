from abc import ABC, abstractmethod
from transformers import AutoConfig
from Classes.TypeDicts import MessagePacket

formatters: dict[str, "Formatter"] = {}


class Formatter(ABC):
    MODEL_NAME: str
    MODEL_LABEL: str
    USER_LABEL: str

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_labels(self) -> list[str]:
        """Returns the labels in the format [USER_LABEL, MODEL_LABEL]

        Returns:
            list[str]: [USER_LABEL, MODEL_LABEL]
        """
        return [self.USER_LABEL, self.MODEL_LABEL]

    @abstractmethod
    def format_train(self, packet: MessagePacket) -> str:
        """Formates a MessagePacket to a finetuneable string

        Args:
            packet (MessagePacket): the packet to format

        Raises:
            NotImplementedError

        Returns:
            str: the formatted packet as a string for finetuning
        """
        raise NotImplementedError

    def format_block(self, block: list[MessagePacket]) -> str:
        """Formats a block of MessagePackets to a string

        Args:
            block (list[MessagePacket]): the block to format

        Returns:
            str: the formatted block as a string
        """
        return "".join(self.format_train(packet) for packet in block)


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
