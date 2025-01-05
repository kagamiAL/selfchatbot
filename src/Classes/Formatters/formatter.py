import torch
from typing import Optional
from warnings import warn
from abc import ABC, abstractmethod
from transformers import AutoConfig, PreTrainedTokenizer
from Classes.TypeDicts import MessagePacket

formatters: dict[str, "Formatter"] = {}


class Formatter(ABC):
    MODEL_NAME: str
    MODEL_LABEL: str
    USER_LABEL: str
    SPECIAL_TOKENS: Optional[list[str]] = None
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_labels(self) -> list[str]:
        """Returns the labels in the format [USER_LABEL, MODEL_LABEL]

        Returns:
            list[str]: [USER_LABEL, MODEL_LABEL]
        """
        return [self.USER_LABEL, self.MODEL_LABEL]

    def add_special_tokens_to_tokenizer(self):
        """Adds special tokens to the tokenizer if they are not already present.

        This method checks if the tokens specified in `SPECIAL_TOKENS` are already
        in the tokenizer's vocabulary. If any tokens are missing, they are added
        as additional special tokens to the tokenizer.

        Returns:
            bool: True if new special tokens were added to the tokenizer, False otherwise.
        """

        if self.SPECIAL_TOKENS:
            comparison = set(self.SPECIAL_TOKENS)
            for token in self.tokenizer.get_vocab():
                if token in comparison:
                    comparison.remove(token)
            if comparison:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": list(comparison)}
                )
                return True
        return False

    @abstractmethod
    def format_block(self, block: list[MessagePacket]) -> str:
        """Formats a block of MessagePackets to a string

        Args:
            block (list[MessagePacket]): the block to format

        Raises:
            NotImplementedError

        Returns:
            str: the formatted block as a string
        """
        raise NotImplementedError

    @abstractmethod
    def get_prompt_encoding(
        self, block: list[MessagePacket]
    ) -> dict[str, torch.Tensor]:
        """Encodes a block of MessagePackets into a dictionary containing the input_ids and attention_mask

        Args:
            block (list[MessagePacket]): The block to encode

        Raises:
            NotImplementedError:

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the input_ids and attention_mask
        """
        raise NotImplementedError

    @abstractmethod
    def filter_output(self, prompt: torch.Tensor, raw_output: torch.Tensor) -> str:
        """Filters the output of the model and returns a string
        Input and output Tensors are NOT padded so they may be different sizes

        Args:
            prompt (torch.Tensor): The input used to generate the output
            raw_output (torch.Tensor): The raw output of the model

        Raises:
            NotImplementedError

        Returns:
            str: The filtered output as a string
        """
        raise NotImplementedError


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
                warn(f"Using formatter for architecture {architecture}")
                return formatters[architecture](tokenizer)
    return formatters["Default"](tokenizer)


def register_formatter(model_name: str, formatter: Formatter):
    """Registers a formatter for a given model name.

    Args:
        model_name (str): The model name to register the formatter for.
        formatter (Formatter): The formatter to register.

    """
    formatters[model_name] = formatter
